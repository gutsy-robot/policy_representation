import json
from policy_representation.utils import scale_action, scale_state
from keras.models import load_model
from policy_representation.utils import trajectory_from_json, get_agent_trajectory
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pickle


"""

This example demonstrates how to generate embeddings from a saved json file and show a pca plot of the 

embeddings from json and compares it to the training trajectories.

"""


# specify the id of the agent for which you wish to generate embeddings for from your json.
# should be either 0 or 1.
# also be careful that you are using the a bait policy representation model for the bait in your json
# and a shooter model for shooter.
agent_id = 1


json_path = '../json/shooter-turn-2-2020-06-05-16-16-56.json'
with open(json_path) as f:
    log = json.load(f)

# load model
model_path = '../models/ae.h5'
model = load_model(model_path)

# extract the combined state and actions.
states, actions = trajectory_from_json(log)

# extract the state, actions for the agent_id(0 or 1) you wish to calculate the embeddings for.
agent_states, agent_actions = get_agent_trajectory(states, actions, agent_id=agent_id)

# scale the state and actions.
agent_states, agent_actions = scale_state(agent_states), scale_action(agent_actions)

use_actions = model.layers[0].input_shape[-1] == 19

if use_actions:
    inp = np.hstack((agent_states, agent_actions))

else:
    inp = agent_states

json_embedding = model.predict(inp.reshape((1, inp.shape[0], inp.shape[1])))


# load stored agent trajectories for comparison.
agent_trajectories = np.load('../trajectories/X_train.npy')
agent_labels = np.load('../trajectories/y_train.npy')

with open("../trajectories/agent_key.txt", "rb") as fp:
    agent_key = pickle.load(fp)

if not use_actions:
    agent_trajectories = agent_trajectories[:, :, :16]

# agent embeddings can be used to find similarity with the json_embedding.
agent_embeddings = model.predict(agent_trajectories)
print("agent embeddings generated successfully")


# for visulisation
fig_pca, ax_pca = plt.subplots()
fig_pca.suptitle('PCA')

pca = PCA(n_components=2)
agent_pca_embeddings = pca.fit_transform(agent_embeddings)

num_agents = len(set(list(agent_labels)))

pca_emb = []
for k in range(num_agents):
    pca_emb.append(agent_embeddings[np.argwhere(agent_labels == k).flatten()])

for j in range(0, len(pca_emb)):
    if agent_key is None:
        ax_pca.scatter(pca_emb[j][:, 0], pca_emb[j][:, 1], label=str(j))
    else:
        ax_pca.scatter(pca_emb[j][:, 0], pca_emb[j][:, 1], label=agent_key[j])

json_pca_emb = pca.transform(json_embedding)
ax_pca.scatter(json_pca_emb[:, 0], json_pca_emb[:, 1], label="json")
ax_pca.legend()
plt.show()
