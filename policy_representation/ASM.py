class ASM():
	def __init__(self):
		model_path = 'models/ae.h5'
		self.model = load_model(model_path)
	def generateEmbedding(self,state_action):
		states=[]
		actions=[]
		for i in range(len(state_action)):
			states.append(state_action[i][0])
			states.append(state_action[i][1])
		agent_states, agent_actions = get_agent_trajectory(states, actions, agent_id=agent_id)
		agent_states, agent_actions = scale_state(agent_states), scale_action(agent_actions)
		use_actions = model.layers[0].input_shape == 19
		if use_actions:
		    inp = np.hstack((agent_states, agent_actions))
		else:
		    inp = agent_states
		return  model.predict(inp.reshape((1, inp.shape[0], inp.shape[1])))
