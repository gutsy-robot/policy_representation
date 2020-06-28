import matplotlib.pyplot as plt
import numpy as np
import os, sys
from ruamel.yaml import YAML
from policy_representation.utils import load_data

from model import Classifier
import torch

import logger
import datetime
import dateutil.tz

yaml = YAML()
v = yaml.load(open(sys.argv[1]))

agent_ids = v['agents']
num_agents = len(agent_ids.split())
print(agent_ids)

root_dir = v['root_dir']
rollout_path = v['rollout_dir']
exp_name = v['experiment_name']
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

torch.set_num_threads(1)
pid=os.getpid()
device = torch.device(f"cuda:{v['cuda']}" if torch.cuda.is_available() and v['cuda'] >= 0 else "cpu")

now = datetime.datetime.now(dateutil.tz.tzlocal())
log_folder = logs_path + f'/classifier/' + now.strftime('%Y_%m_%d_%H_%M_%S')
logger.configure(dir=log_folder)
os.makedirs(log_folder +'/models', exist_ok=True)
os.system(f'cp src/train_agent_classifier.py {log_folder}')
os.system(f'cp {sys.argv[1]} {log_folder}/variant_{pid}.yml')

classifier = Classifier(classes=num_agents, device=device, **v['model'])
classifier.learn(X_train, y_train, X_test, y_test, epoch=v['epoch'], batch_size=v['batch_size'])


