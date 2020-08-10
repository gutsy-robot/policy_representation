import json
from .policy_representation.utils import scale_action, scale_state
from keras.models import load_model
from .policy_representation.utils import trajectory_from_json, get_agent_trajectory
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pickle

class ASM():
	def __init__(self, role = 'S'):
		if role=='S':
			model_path = '/home/suhas/webtsf/server/agents/policy_representation/models/bait_identifier.h5'
		else:
			model_path = '/home/suhas/webtsf/server/agents/policy_representation/models/shooter_identifier0.h5'
		self.model = load_model(model_path)
	def generateEmbedding(self,states):
		trajectory_json = {"game_states":states}
		states, actions = trajectory_from_json(trajectory_json)
		# extract the state, actions for the agent_id(0 or 1) you wish to calculate the embeddings for.
		agent_states, agent_actions = get_agent_trajectory(states, actions, agent_id=1) #todo make sure this is right
		# scale the state and actions.
		agent_states, agent_actions = scale_state(agent_states), scale_action(agent_actions)
		use_actions = self.model.layers[0].input_shape[-1] == 19
		if use_actions:
		    inp = np.hstack((agent_states, agent_actions))
		else:
		    inp = agent_states
		embedding = self.model.predict(inp.reshape((1, inp.shape[0], inp.shape[1])))
		return embedding

