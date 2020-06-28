import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import time
from utils import logger

class Classifier(nn.Module):
    def __init__(self, input_dim=19, hidden_dim=8, bidirectional=True, num_layers=2, arch='GRU',
                classes=9, lr=0.001, device=torch.device('cpu')):
        super().__init__()
        self.device = device
        self.classes = classes
        self.hidden_dim = hidden_dim
        self.direction = 2 if bidirectional else 1
        assert arch in ['LSTM', 'GRU']
        RNN = nn.LSTM if arch == 'LSTM' else nn.GRU
        self.lstm = RNN(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, 
                            bidirectional=bidirectional).to(device)
        self.linear = nn.Linear(self.direction*hidden_dim, classes).to(device)

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(list(self.lstm.parameters()) + list(self.linear.parameters()),
                                    lr=lr)

    def forward(self, input):
        # input (B, T, S) -> x (T, B, S)
        # h0, c0 (2*L, B, H) are default zeros
        x = input.transpose(0, 1)
        o, _ = self.lstm(x)
        # output (T, B, 2*H), hn, cn (2*L, B, H)
        o = F.dropout(o)
        o = o.transpose(0, 1).contiguous().view(-1, self.direction*self.hidden_dim) # (B*T, 2*H)
        logits = self.linear(o) # (B*T, C)
        return logits
    
    def learn_single(self, input, label, train=True):
        input = torch.FloatTensor(input).to(self.device)
        B, T, S = input.shape
        label = torch.LongTensor(label).to(self.device) # (B, )
        label = label.unsqueeze(1).expand(-1, T).contiguous().view(-1) # (B*T,)

        logits = self.forward(input) # (B*T, C)
        loss = self.criterion(logits, label)

        if train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        _, pred = torch.max(logits, 1)
        acc = (pred == label).float().mean()

        return loss.item(), acc.item()

    def learn(self, X_train, y_train, X_test, y_test, epoch, batch_size):
        num_train, num_test = y_train.shape[0], y_test.shape[0]
        print("start training!")
        history_test_acc = 0.0

        for e in range(epoch):
            
            start = time.time()

            metrics = {'tr_loss': [], 'tr_acc': [], 'ts_loss': [], 'ts_acc': []}

            indices_train = list(range(num_train))
            random.shuffle(indices_train)
            indices_test = list(range(num_test))
            random.shuffle(indices_test)

            for i in range(num_train // batch_size):
                ind = indices_train[i*batch_size: (i+1)*batch_size]
                loss, acc = self.learn_single(X_train[ind], y_train[ind])
                metrics['tr_loss'].append(loss)
                metrics['tr_acc'].append(acc)

            for i in range(num_test // batch_size):
                ind = indices_test[i*batch_size: (i+1)*batch_size]
                loss, acc = self.learn_single(X_test[ind], y_test[ind], train=False)
                metrics['ts_loss'].append(loss)
                metrics['ts_acc'].append(acc)
            
            logger.record_tabular('Epoch', e)
            logger.record_tabular('Train Loss', round(np.mean(metrics['tr_loss']), 3))
            logger.record_tabular('Train Acc', round(np.mean(metrics['tr_acc']), 3))
            logger.record_tabular('Test Loss', round(np.mean(metrics['ts_loss']), 3))
            logger.record_tabular('Test Acc', round(np.mean(metrics['ts_acc']), 3))
            logger.dump_tabular()

            test_acc = np.mean(metrics['ts_acc'])
            if history_test_acc < test_acc and e > 10:
                history_test_acc = test_acc
                print("new record", history_test_acc)
                torch.save(self.state_dict(), f'{logger.get_dir()}/models/e{e}_acc{test_acc*100:.0f}.pt')
    
    def infer(self, X_test):
        X_test = torch.FloatTensor(X_test).to(self.device)
        B, T, S = X_test.shape
        logits = self.forward(X_test).reshape(B, T, -1) # (B*T, C)
        probs = F.softmax(logits, dim=-1) # (B, T, C) entire trajectory
        return probs.cpu().detach().numpy()
