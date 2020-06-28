import os, sys
import subprocess
from itertools import product
import time
from ruamel.yaml import YAML
from pathlib import Path
yaml = YAML()
yaml.default_flow_style = None # https://stackoverflow.com/a/56939573/9072850

configs = [
            "configs/bait_classifier.yml",
          ]

programs = {
            "classifier": "src/train_agent_classifier.py", 
        }

epoch = 100
batch_size = 200
hidden_dim = 8 
bis = [False, True]
num_layers = [1, 2]
archs = ['LSTM', 'GRU']
lrs = [0.01, 0.001]

cudas = (4, 7)
cuda_num = cudas[1] - cudas[0] + 1

for idx, (config, arch, bi, num_layer, lr) in enumerate(product(configs, archs, bis, num_layers, lrs)):
    if "classifier" in config:
        program = programs["classifier"]

    v = yaml.load(open(config))

    v['cuda'] = idx % cuda_num + cudas[0]

    v['model']['hidden_dim'] = hidden_dim
    v['model']['bidirectional'] = bi
    v['model']['num_layers'] = num_layer
    v['model']['arch'] = arch
    v['model']['lr'] = lr

    v['batch_size'] = batch_size
    v['epoch'] = epoch


    yaml.dump(v, Path(config))

    command = f"python {program} {config}"
    print(idx, command)
    
    with open(os.devnull, 'w') as f:
        proc = subprocess.Popen(command, shell=True, stdout=f)
    time.sleep(6)






# ==============

# # train autoencoder with auxiliary action prediction, with only the state sequence as the input.
# os.system('python -m scripts.train_autoencoder --train  '
#           '--experiment_name ' + experiment_name +
#           ' --data_path ' + data_dir + ' --plots_path ' + plots_dir +
#           ' --use_auxiliary_pred --epochs ' + str(epochs) + ' --num_agents ' + str(num_agents))

# # train autoencoder with auxiliary classifier as well as action predictor using only state sequence as input.
# os.system('python -m scripts.train_autoencoder --train  '
#           '--experiment_name ' + experiment_name + ' --num_agents ' + str(num_agents) +
#           ' --data_path ' + data_dir + ' --plots_path ' + plots_dir +
#           ' --use_classifier --use_auxiliary_pred --epochs ' + str(epochs))

# # train a standard autoencoder using both the state and actions as input.
# os.system('python -m scripts.train_autoencoder --train  '
#           '--experiment_name ' + experiment_name + ' --num_agents ' + str(num_agents) +
#           ' --data_path ' + data_dir + ' --plots_path ' + plots_dir +
#           ' --use_actions --epochs ' + str(epochs))

# # train a standard autoencoder using both the state and actions as input and an auxiliary classifier.
# os.system('python -m scripts.train_autoencoder --train '
#           '--experiment_name ' + experiment_name + ' --num_agents ' + str(num_agents) +
#           ' --data_path ' + data_dir + ' --plots_path ' + plots_dir +
#           ' --use_actions --use_classifier --epochs ' + str(epochs))

# # path to your json file.
# json = 'shooter-turn-2-2020-06-05-16-16-56.json'

# os.system('python -m scripts.encode_json '
#           '--experiment_name ' + experiment_name + ' --num_agents ' + str(num_agents) +
#           ' --data_path ' + data_dir + ' --plots_path ' + plots_dir +
#           ' --use_actions  --num_agents ' + str(num_agents) +
#           ' --agent_id 1 --embedding_generation_interval 10'
#           ' --json_path ' + json)
