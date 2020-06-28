import numpy as np
import os, sys
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
from ruamel.yaml import YAML
from utils import logger
from utils.data import *
import datetime
import dateutil.tz

from src.model import Classifier
import torch

yaml = YAML()
v = yaml.load(open(sys.argv[1]))

agent_ids = v['agents']
num_agents = len(agent_ids.split())
print(agent_ids)

root_dir = v['root_dir']
rollout_path = v['rollout_dir']
exp_name = v['experiment_name']

torch.set_num_threads(1)
pid=os.getpid()
np.set_printoptions(precision=3, suppress=True)
device = torch.device(f"cuda:{v['cuda']}" if torch.cuda.is_available() and v['cuda'] >= 0 else "cpu")

if v['mode'] != 'human':
    data_path = os.path.join(root_dir, 'data', exp_name)
    logs_path = os.path.join(root_dir, 'logs', exp_name)
    os.makedirs(logs_path, exist_ok=True)

    generate = v['generate']
    load_baits = v['load_baits']

    if generate:
        X_train, X_test, y_train, y_test = load_data(agent_ids, rollout_path, train_on_states=False, load_baits=load_baits)

        np.save(data_path + '/X_train.npy', X_train)
        np.save(data_path + '/y_train.npy', y_train)

        np.save(data_path + '/X_test.npy', X_test)
        np.save(data_path + '/y_test.npy', y_test)

        # agent_key = []
        for k in range(len(agent_ids)):
            id = agent_ids[k]
            if load_baits:
                bait_id = id.split('Bait_')[1]
                # agent_key.append('Bait_' + bait_id)

            else:
                shooter_id = id.split('_Bait_')[0]
                # agent_key.append(shooter_id)

        # with open(data_path + "/agent_key.txt", "wb") as fp:
        #     pickle.dump(agent_key, fp)
        exit()

    X_train = np.load(data_path + '/X_train.npy')
    X_test = np.load(data_path + '/X_test.npy')
    y_train = np.load(data_path + '/y_train.npy')
    y_test = np.load(data_path + '/y_test.npy')

    print("shape of X_train: ", X_train.shape)
    print("shape of y_train: ", y_train.shape)
    print("shape of X_test: ", X_test.shape)
    print("shape of y_test: ", y_test.shape)

    if v['mode'] == 'train':
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        log_folder = logs_path + f'/classifier/' + now.strftime('%Y_%m_%d_%H_%M_%S')
        logger.configure(dir=log_folder)
        os.makedirs(log_folder +'/models', exist_ok=True)
        os.system(f'cp src/train_agent_classifier.py {log_folder}')
        os.system(f'cp {sys.argv[1]} {log_folder}/variant_{pid}.yml')

        classifier = Classifier(classes=num_agents, device=device, **v['model'])
        classifier.learn(X_train, y_train, X_test, y_test, epoch=v['epoch'], batch_size=v['batch_size'])

    elif v['mode'] == 'evaluate':
        classifier = Classifier(classes=num_agents, device=device, **v['model'])
        classifier.load_state_dict(torch.load(os.path.join(v['model_path'])))
        probs = classifier.infer(X_test)

        B, T, S = X_test.shape
        acc = []
        for t in range(T):  
            acc.append((np.argmax(probs[:, t, :], 1) == y_test).mean())
        plt.plot(acc)
        plt.title('accuracy vs timestamp')
        plt.xlabel('timestamp')
        plt.ylabel('avg accuracy')
        plt.savefig(f'imgs/logs/{exp_name}/acc_time.pdf')
        plt.close()

        sim = np.zeros((num_agents, num_agents))
        cnt = np.zeros((num_agents, 1))
        for n in range(B):
            sim[y_test[n]] += probs[n, 1200] # the ending is less accurate?
            cnt[y_test[n]] += 1
        sim /= cnt
        print(sim)

else:
    human_path = "/home/tianwei/webtsf/server/disk/Yikang/turn_only_shooter_35/speed15"
    traj = parseAllJson(human_path, 0)
    human_states, human_actions = [], []
    for (state, action) in traj:
        states, actions = get_human_trajectory(state, action, True)
        human_states.append(states)
        human_actions.append(actions)
    human_states = np.array(human_states)
    human_actions = np.array(human_actions)

    human_data = np.concatenate([human_states, human_actions], axis=-1)

    classifier = Classifier(classes=num_agents, device=device, **v['model'])
    classifier.load_state_dict(torch.load(v['model_path'], map_location=device))
    probs = classifier.infer(human_data)

    B, T, C = probs.shape
    for i in range(B):
        print(probs[i, :].mean(0))

    # plot time curves
    for b in range(B):
        for i in range(num_agents):
            data = uniform_filter(probs[b, :, i], 100)
            plt.plot(data, label=i)
            plt.text(np.argmax(data), np.max(data), str(i))
        plt.xlabel('timestamp')
        plt.ylabel('probs')
        plt.show()
        # plt.savefig(f'imgs/logs/{exp_name}/acc_time.pdf')
        plt.close()
    