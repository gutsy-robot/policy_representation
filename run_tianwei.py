import os

# specify the directory where you wish to store your trained models and training data.
data_dir = '/serverdata/tianwei/policy_representation/data/'

# specify the directory where your training plots and pca plots would be stored.
plots_dir = '/serverdata/tianwei/policy_representation/plots/'

# name of the experiment. You can choose any name.
experiment_name = 'all_baits_Turn_Only_Shooter5'
# experiment_name = 'all_shooters_Bait_yikang0'

# specify the path of your rollout directory
rollout_dir = '/serverdata/tianwei/self_play_trajectories/'

# directories inside rollout dir inside which you have your bait_states.npy etc.
# # These should be equal to num_agents.
agents = 'Turn_Only_Shooter5_Bait_Sid7 ' \
         'Turn_Only_Shooter5_Bait_Yikang0 Turn_Only_Shooter5_Bait_Sid8 Turn_Only_Shooter5_Bait_Yikang1 ' \
         'Turn_Only_Shooter5_Bait_Yikang2 Turn_Only_Shooter5_Bait_Yikang3 Turn_Only_Shooter5_Bait_Yikang4 ' \
         'Turn_Only_Shooter5_Bait_Yikang5 Turn_Only_Shooter5_Bait_Yikang6'

# agents = 'Turn_Only_Shooter2_Bait_Yikang0 ' \
#          'Turn_Only_Shooter3_Bait_Yikang0 Turn_Only_Shooter4_Bait_Yikang0 Turn_Only_Shooter5_Bait_Yikang0 ' \
#          'Turn_Only_Shooter6_Bait_Yikang0 Mirror_Shooter1_Bait_Yikang0 Mirror_Shooter2_Bait_Yikang0 ' \
#          'Mirror_Shooter3_Bait_Yikang0'

# number of different agents for the classifier model.
num_agents = len(agents.split())

# path to your json file.
json = 'shooter-turn-2-2020-06-05-16-16-56.json'
epochs = 20
# ==============

# generate training data for baits
os.system('python scripts/train_agent_classifier.py --load_baits --data_path ' + data_dir +
          ' --experiment_name ' + experiment_name + ' --plots_path ' + plots_dir +
          ' --generate_data --num_agents ' + str(num_agents) + ' --rollout_path ' + rollout_dir +
          ' --agents ' + agents)
exit()
# # generate training data for shooters
# os.system('python scripts/train_agent_classifier.py --data_path ' + data_dir +
#           ' --experiment_name ' + experiment_name + ' --plots_path ' + plots_dir +
#           ' --generate_data --num_agents ' + str(num_agents) + ' --rollout_path ' + rollout_dir +
#           ' --agents ' + agents)

# # train a classification model.
# os.system('python -m scripts.train_agent_classifier --epochs ' + str(epochs) + ' --num_agents '
#           + str(num_agents) +
#           ' --experiment_name ' + experiment_name + ' --train '
#                                                     '--data_path ' + data_dir + ' --plots_path ' + plots_dir +
#           ' --agents ' + agents)

# train autoencoder with auxiliary action prediction, with only the state sequence as the input.
os.system('python -m scripts.train_autoencoder --train  '
          '--experiment_name ' + experiment_name +
          ' --data_path ' + data_dir + ' --plots_path ' + plots_dir +
          ' --use_auxiliary_pred --epochs ' + str(epochs) + ' --num_agents ' + str(num_agents))

# train autoencoder with auxiliary classifier as well as action predictor using only state sequence as input.
os.system('python -m scripts.train_autoencoder --train  '
          '--experiment_name ' + experiment_name + ' --num_agents ' + str(num_agents) +
          ' --data_path ' + data_dir + ' --plots_path ' + plots_dir +
          ' --use_classifier --use_auxiliary_pred --epochs ' + str(epochs))

# train a standard autoencoder using both the state and actions as input.
os.system('python -m scripts.train_autoencoder --train  '
          '--experiment_name ' + experiment_name + ' --num_agents ' + str(num_agents) +
          ' --data_path ' + data_dir + ' --plots_path ' + plots_dir +
          ' --use_actions --epochs ' + str(epochs))

# train a standard autoencoder using both the state and actions as input and an auxiliary classifier.
os.system('python -m scripts.train_autoencoder --train '
          '--experiment_name ' + experiment_name + ' --num_agents ' + str(num_agents) +
          ' --data_path ' + data_dir + ' --plots_path ' + plots_dir +
          ' --use_actions --use_classifier --epochs ' + str(epochs))
#

os.system('python -m scripts.encode_json '
          '--experiment_name ' + experiment_name + ' --num_agents ' + str(num_agents) +
          ' --data_path ' + data_dir + ' --plots_path ' + plots_dir +
          ' --use_actions  --num_agents ' + str(num_agents) +
          ' --agent_id 1 --embedding_generation_interval 10'
          ' --json_path ' + json)
