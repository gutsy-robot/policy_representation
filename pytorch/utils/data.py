from heapq import heappop, heappush, heappushpop
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import os
import json

def dist_from_player(player_pos, projectile_pos):
    return np.linalg.norm(player_pos - projectile_pos)

def getStateDict_Json(json_data):
    state_dict = {'players': [],
                  'fortresses': [],
                  'shells': [],
                  'missiles': [],
                  'actions': []
                  }

    # Extract player data
    for j in range(2):
        player = json_data["players"][str(j)]
        state_dict['players'].append((str(j), player["position"]["x"], player["position"]["y"],
                                      player["velocity"]["x"],
                                      player["velocity"]["y"], player["angle"], player["alive"]))
        state_dict['actions'].append(
            [player["action"]["turn"], int(player["action"]["thrust"]), int(player["action"]["fire"])])  # must parse it

    for j in range(1):
        fortress = json_data["fortresses"][str(j)]
        state_dict['fortresses'].append((str(j), fortress["activationRegion"]["position"]["x"],
                                         fortress["activationRegion"]["position"]["y"],
                                         fortress["angle"], fortress["alive"], fortress["shield"]["vulnerable"],
                                         fortress["target"], fortress["shield"]["radius"],
                                         fortress["activationRegion"]["radius"]))
    if json_data["shells"] is not None:
        # print("shells: ", json_data["shells"])
        for k, shell in json_data["shells"].items():
            state_dict['shells'].append((k, shell["position"]["x"], shell["position"]["y"], shell["angle"]))
            # print((k, shell["position"]["x"], shell["position"]["y"], shell["angle"]))
    if json_data["missiles"] is not None:
        # print(json_data["missiles"])
        for k, missile in json_data["missiles"].items():
            state_dict['missiles'].append((k, missile["position"]["x"], missile["position"]["y"], missile["angle"]))

    return state_dict

def getStateArray(state_dict, bait_id, shells=True):
    state_array = -1 * np.ones((28,), np.float)
    state_array[:4] = np.array([
        state_dict['fortresses'][0][3],
        state_dict['fortresses'][0][4],
        state_dict['fortresses'][0][6],
        state_dict['fortresses'][0][5]
    ])
    state_array[4:16] = np.array([
        state_dict['players'][0][1],
        state_dict['players'][0][2],
        state_dict['players'][0][5],
        state_dict['players'][0][3],
        state_dict['players'][0][4],
        state_dict['players'][0][6],
        state_dict['players'][1][1],
        state_dict['players'][1][2],
        state_dict['players'][1][5],
        state_dict['players'][1][3],
        state_dict['players'][1][4],
        state_dict['players'][1][6],
    ])

    if shells:  # default: Yikang bait, two nearest shells to the bait
        bait_pos = state_array[bait_id * 6 + 4: bait_id * 6 + 6]
        heap = []
        for (_, s_x, s_y, s_a) in state_dict['shells']:
            if len(heap) < 2:
                heappush(heap, (-dist_from_player(bait_pos, np.array([s_x, s_y])), s_x, s_y, s_a))
            else:
                heappushpop(heap, (-dist_from_player(bait_pos, np.array([s_x, s_y])), s_x, s_y, s_a))
        if heap:
            state_array[16:19] = heappop(heap)[1:4]
        else:
            state_array[16:19] = [0, 1, 90]
        if heap:
            state_array[19:22] = heappop(heap)[1:4]
        else:
            state_array[19:22] = [0, 1, 90]

    else:  # nearest shell to bait and shooter
        for (_, s_x, s_y, s_a) in state_dict['shells']:
            if state_array[18] == -1 or \
                    dist_from_player(state_array[4:6], np.array([s_x, s_y])) < dist_from_player(state_array[4:6],
                                                                                                state_array[16:18]):
                state_array[16:19] = np.array([s_x, s_y, s_a])
            if state_array[21] == -1 or \
                    dist_from_player(state_array[10:12], np.array([s_x, s_y])) < dist_from_player(state_array[10:12],
                                                                                                  state_array[19:21]):
                state_array[19:22] = np.array([s_x, s_y, s_a])

    for (_, s_x, s_y, s_a) in state_dict['missiles']:
        if state_array[24] == -1 or \
                dist_from_player(state_array[4:6], np.array([s_x, s_y])) < dist_from_player(state_array[4:6],
                                                                                            state_array[22:24]):
            state_array[22:25] = np.array([s_x, s_y, s_a])
        if state_array[27] == -1 or \
                dist_from_player(state_array[10:12], np.array([s_x, s_y])) < dist_from_player(state_array[10:12],
                                                                                              state_array[25:27]):
            state_array[25:28] = np.array([s_x, s_y, s_a])

    action_array = np.concatenate(state_dict['actions'])
    action_key = {'no_turn' : 0, 'turn_left' : 1, 'turn_right': 2}

    actions = [action_key[action_array[0]],
               bool(action_array[1]),
               bool(action_array[2]),
               action_key[action_array[3]],
               bool(action_array[4]),
               bool(action_array[5])]
    return state_array, actions

def trajectory_from_json(log, bait_id, shells=True):
    ep_states, ep_actions = [], []

    for state in log['game_states']:
        json_data = getStateDict_Json(state)
        states, actions = getStateArray(json_data, bait_id, shells)
        ep_states.append(states)
        ep_actions.append(actions)

    ep_states = np.array(ep_states)
    ep_actions = np.array(ep_actions)

    return ep_states, ep_actions

def parseAllJson(data_path: str, bait_id: int):
	# data_path: absolute path
    json_list = sorted(os.listdir(data_path))
    trajs = []
    for json_name in json_list:
        if json_name[-5:] != '.json': continue
        with open(os.path.join(data_path, json_name)) as f:
        	json_data = json.load(f)
        state, action = trajectory_from_json(json_data, bait_id=bait_id, shells=True)
        trajs.append((state, action))
        print(json_name, "shape", state.shape, action.shape)

    return trajs

def get_human_trajectory(episode_states, episode_actions, bait: bool):
    # assert human_id=0
    if bait:
        human_states = np.concatenate((episode_states[:, :10], 
                                       episode_states[:,  16:22]), axis=1) # shells
    else:
        human_states = np.concatenate((episode_states[:, :10], 
                                       episode_states[:, 22:28]), axis=1) # missiles

    human_actions = episode_actions[:, :3]

    human_states = scale_state(human_states)
    human_actions = scale_action(human_actions)

    return human_states, human_actions

def load_data(agent_ids, agents_path, train_on_states_only=False, load_baits=True):
    """
    agent_ids: name of bait/shooter policies
    train_on_states_only: Whether to train on states only or both states and actions
    baits_path: Common Directory for bait trajectories

    """
    num_agents = len(agent_ids)
    train_x = []
    train_y = []

    # for figuring how much zero padding needs to be added.
    max_steps = 0

    # store actual length of each episode to get back the actual length sequences after scaling.
    episode_lengths = []

    for i in range(0, len(agent_ids)):
        # load training data
        agent = agent_ids[i]
        if not load_baits:
            actions_path = agents_path + agent + '/shooter_actions.npy'
            state_path = agents_path + agent + '/shooter_states.npy'

        else:
            actions_path = agents_path + agent + '/bait_actions.npy'
            state_path = agents_path + agent + '/bait_states.npy'

        actions = np.load(actions_path, allow_pickle=True)
        states = np.load(state_path, allow_pickle=True)

        assert len(states) == len(actions)
        for j in range(0, len(actions)):
            episode_states = scale_state(np.array(states[j]))
            episode_actions = scale_action(np.array(actions[j]))
            num_steps = len(episode_actions)

            episode_lengths.append(num_steps)

            # if the number of steps is more than current max.
            if num_steps > max_steps:
                max_steps = num_steps
            assert len(episode_actions) == len(episode_states)

            if not train_on_states_only:
                combined_x = np.hstack((episode_states, episode_actions))
            else:
                combined_x = episode_states

            train_x.append(combined_x)
            train_y.append(i)

    train_x, train_y = np.array(train_x), np.array(train_y)
    print("all data", train_x.shape, train_y.shape)

    # shuffle data
    X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def scale_state(trajectory, x_min=-355.0, x_max=355.0,
                y_min=-310.0, y_max=310.0, max_speed=180.0, angle_max=360.0, target_max=1.0):
    """

    trajectory: (timesteps, bait/shooter state)
    x_min: Min x coordinate in the tsf game
    x_max: Max x
    y_min: Min y
    y_max: Max y

    max_speed: Max speed in the tsf game.

    scale to [0, 1] but target in [-1, 1]?
    """
    assert len(trajectory.shape) == 2
    angle_indices = [0, 6, 12, 15]
    x_indices = [4, 10, 13]
    y_indices = [5, 11, 14]
    velocity_indices = [7, 8]
    target_index = [2]

    trajectory[:, target_index] = (trajectory[:, target_index] - (-target_max)) / (2 * target_max)

    # print("x_max: ", np.max(trajectory[:, x_indices]))
    # print("y_max: ", np.max(trajectory[:, x_indices]))
    trajectory[:, angle_indices] = (trajectory[:, angle_indices] - (-angle_max)) / (2 * angle_max)
    trajectory[:, velocity_indices] = (trajectory[:, velocity_indices] - (-max_speed)) / (2 * max_speed)
    # trajectory[:, velocity_indices] /= max_speed
    # print("x_max: ", np.max(trajectory[:, x_indices] - x_min))
    # print("x_max: ", np.max((trajectory[:, x_indices] - x_min) / (x_max - x_min)))

    trajectory[:, x_indices] = (trajectory[:, x_indices] - x_min) / (x_max - x_min)
    trajectory[:, y_indices] = (trajectory[:, y_indices] - y_min) / (y_max - y_min)

    assert np.max(trajectory[:, target_index]) <= 1 and np.min(trajectory[:, target_index]) >= 0

    assert np.max(trajectory[:, angle_indices]) <= 1 and np.min(trajectory[:, angle_indices]) >= 0
    assert np.max(trajectory[:, velocity_indices]) <= 1 and np.min(trajectory[:, velocity_indices]) >= 0
    # print(np.max(trajectory[:, x_indices]), np.min(trajectory[:, x_indices]))

    assert np.max(trajectory[:, x_indices]) <= 1 and np.min(trajectory[:, x_indices]) >= 0
    assert np.max(trajectory[:, y_indices]) <= 1 and np.min(trajectory[:, y_indices]) >= 0

    return trajectory

def scale_action(trajectory):
    # scale to [0, 1]
    assert len(trajectory.shape) == 2
    # print(trajectory.shape)
    # print(trajectory[:, 0])

    trajectory = trajectory.astype('float')
    trajectory[:, 0] *= 0.5

    return trajectory
