from keras.models import load_model
#from policy_representation.utils import scale_action, scale_state
#from policy_representation.utils import trajectory_from_json, get_agent_trajectory
import numpy as np
import pickle as pk

folder =  "baits_gdr/"

x_path = folder+"trajectories/"+"X_train.npy"
y_path = folder+"trajectories/"+"y_train.npy"
key_path = folder+"trajectories/"+"agent_key.txt"
model_path = folder+"models/"+"encoder_aux_pred.h5"
dump_name = folder[:-1] + "_embedding_dict.pickle"


model = load_model(model_path)
agent_trajectories = np.load(x_path)
agent_labels = np.load(y_path)
agent_key = pk.load(open(key_path,'rb'))

use_actions = model.layers[0].input_shape[-1] == 19

if not use_actions:
    agent_trajectories = agent_trajectories[:, :, :16]

out = dict()
agent_embeddings = model.predict(agent_trajectories)

for i in range(len(agent_key)):
	key = agent_key[i]
	out[key]=list()
for j in range(agent_labels.shape[0]):
	out[agent_key[agent_labels[j]]].append(agent_embeddings[j,:])
with open(dump_name,'wb') as fp:
	pk.dump(out,fp)
