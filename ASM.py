import json
from .policy_representation.utils import scale_action, scale_state
from .policy_representation.utils import trajectory_from_json, get_agent_trajectory
from keras.models import load_model
import numpy as np
import pickle

shooter_model = load_model('policy_representation/models/bait_identifier.h5')
bait_model = load_model('policy_representation/models/shooter_identifier0.h5')

class ASM():
	def __init__(self, role = 'S'):
		if role=='S':
			self.model = shooter_model
		else:
			self.model = bait_model
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

