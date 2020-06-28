from keras.layers import Input
from keras.models import Model
from keras.layers import Lambda
from keras.layers import RepeatVector
from keras import backend as K
from heapq import heappop, heappush, heappushpop
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, TimeDistributed, Dropout, Bidirectional
from keras.callbacks import ModelCheckpoint
import argparse, os
from keras.optimizers import Adam
import pickle


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


def get_opts():
    parser = argparse.ArgumentParser(description='TSF_RL: Space Fortress')
    parser.add_argument('--use_auxiliary_pred', action="store_true", default=False, help='use auxiliary predictions')
    parser.add_argument('--experiment_name', type=str, default="classification_baits", help='Experiment Name')
    parser.add_argument('--use_actions', action="store_true", default=False, help='use actions')
    parser.add_argument('--use_classifier', action="store_true", default=False, help='use classifier')
    parser.add_argument('--num_agents', type=int, default=2, help='Number of baits')
    parser.add_argument('--epochs', type=int, default=10, help='Number of baits')
    parser.add_argument('--data_path', type=str, default="data/",
                        help='data_path')
    parser.add_argument('--plots_path', type=str, default="plots/",
                        help='data_path')

    parser.add_argument('--train', action="store_true", default=False, help='train')

    opts = parser.parse_args()
    return opts


# custom layer for creating repeating encodings for the decoder to decode to appropriate length.
def repeat_vector(args):
    layer_to_repeat = args[0]
    sequence_layer = args[1]
    return RepeatVector(K.shape(sequence_layer)[1])(layer_to_repeat)


# generator to generate variable length inputs for the autoencoder.
def train_generator(X_train, y_train, actions=None, num_agents=2, use_auxiliary_pred=False, use_classifier=False):
    while True:
        ind = np.random.randint(len(X_train))
        y_gen = X_train[ind].reshape((1, X_train[ind].shape[0], X_train[ind].shape[1]))
        # print("shape of y_gen: ", y_gen.shape)
        if use_auxiliary_pred:
            # print("shape is: ", actions[ind].shape)
            act = actions[ind].reshape((1, X_train[ind].shape[0], 3))
            # print("shape of actions is: ", act.shape)

        x_gen = X_train[ind]
        x_gen = x_gen.reshape((1, x_gen.shape[0], X_train[ind].shape[1]))

        if use_auxiliary_pred:
            if not use_classifier:
                yield x_gen, {'decoder': y_gen, 'action_predictor': act}

            else:
                # y = np.zeros((1, 2))
                y = np.zeros((1, num_agents))
                # print(y_train[ind])
                y[:, y_train[ind]] = 1.0
                # y = y.reshape((1, y.shape[0], y.shape[1]))
                yield x_gen, {'decoder': y_gen, 'action_predictor': act, 'classification': y}

        else:
            if not use_classifier:
                yield x_gen, {'decoder': y_gen}
            else:
                y = np.zeros((1, num_agents))
                y[:, y_train[ind]] = 1.0
                yield x_gen, {'decoder': y_gen, 'classification': y}


def val_generator(X_test, y_test, actions_test=None, num_agents=2, use_auxiliary_pred=False,
                  use_classifier=False):
    while True:
        ind = np.random.randint(len(X_test))
        y_gen = X_test[ind].reshape((1, X_test[ind].shape[0], X_test[ind].shape[1]))
        # print("shape of y_gen: ", y_gen.shape)
        if use_auxiliary_pred:
            act = actions_test[ind].reshape((1, X_test[ind].shape[0], 3))
            # print("shape of actions is: ", act.shape)

        x_gen = X_test[ind]
        x_gen = x_gen.reshape((1, x_gen.shape[0], x_gen.shape[1]))

        if use_auxiliary_pred:
            if not use_classifier:
                yield x_gen, {'decoder': y_gen, 'action_predictor': act}

            else:
                # y = np.zeros((1, 2))
                y = np.zeros((1, num_agents))

                # print(y_train[ind])
                y[:, y_test[ind]] = 1.0
                # y = y.reshape((1, y.shape[0], y.shape[1]))
                yield x_gen, {'decoder': y_gen, 'action_predictor': act, 'classification': y}

        else:
            if not use_classifier:
                yield x_gen,  {'decoder': y_gen}

            else:
                y = np.zeros((1, num_agents))
                y[:, y_test[ind]] = 1.0
                yield x_gen, {'decoder': y_gen, 'classification': y}


def get_model(use_auxiliary_pred=True, use_actions=False, use_classifier=False, num_agents=2):
    # define the model.
    if not use_actions:
        inp = Input(shape=(None, 16), name='input')

    else:
        inp = Input(shape=(None, 19), name='input')

    output_1 = LSTM(12, name='encoder')(inp)
    # output_1 = Dropout(0.5)(output_1)
    output_2 = Lambda(repeat_vector, output_shape=(None, 12))([output_1, inp])

    if not use_actions:
        predictions = TimeDistributed(Dense(16, activation='tanh'), name='decoder')(output_2)

    else:
        predictions = TimeDistributed(Dense(19, activation='tanh'), name='decoder')(output_2)

    if use_classifier:
        softmax_out = Dense(num_agents, activation='softmax', name='classification')(output_1)

    if use_auxiliary_pred:
        action_predictions = TimeDistributed(Dense(3, activation='tanh'), name='action_predictor')(output_2)
        if not use_classifier:
            model = Model(inputs=inp, outputs=[predictions, action_predictions])
            model.compile(optimizer='adam', loss={'decoder': 'mean_squared_error',
                                                  'action_predictor': 'mean_squared_error'})

        else:
            model = Model(inputs=inp, outputs=[predictions, action_predictions, softmax_out])
            model.compile(optimizer='adam', loss={'decoder': 'mean_squared_error',
                                                  'action_predictor': 'mean_squared_error',
                                                  'classification': 'categorical_crossentropy'})

    else:
        if not use_classifier:
            model = Model(inputs=inp, outputs=[predictions])
            model.compile(optimizer='adam', loss={'decoder': 'mean_squared_error'})
        else:
            model = Model(inputs=inp, outputs=[predictions, softmax_out])
            model.compile(optimizer='adam', loss={'decoder': 'mean_squared_error',
                                                  'classification': 'categorical_crossentropy'})
    return model


def train(model, X_train, y_train, actions, X_test, y_test, actions_test, data_path, experiment_name, num_agents=2,
          use_classifier=False, use_auxiliary_pred=False, epochs=1, suffix=''):

    model_name = '/model_ae'
    model_name += suffix
    model_name += '.hdf5'
    # print("model name is: ", model_name)
    checkpoint = ModelCheckpoint(data_path + experiment_name + model_name,
                                 monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='min')
    history = model.fit_generator(train_generator(X_train, y_train, actions=actions,
                                                  num_agents=num_agents, use_classifier=use_classifier,
                                                  use_auxiliary_pred=use_auxiliary_pred),
                                  validation_data=val_generator(X_test, y_test, num_agents=num_agents,
                                                                actions_test=actions_test,
                                                                use_classifier=use_classifier,
                                                                use_auxiliary_pred=use_auxiliary_pred),
                                  validation_steps=len(X_test),
                                  steps_per_epoch=30, epochs=epochs, verbose=1, callbacks=[checkpoint])
    return history


def prepare_data(X_train, y_train, X_test, y_test, num_baits, use_actions=False):
    examples_train = []
    for k in range(num_baits):
        examples_train.append(X_train[np.argwhere(y_train == k).flatten()])

    actions = []
    states = []
    for ex in X_train:
        if not use_actions:
            states.append(ex[:, :16])
            actions.append(ex[:, 16:])

        else:
            states.append(ex)
            actions.append(ex[:, 16:])

    X_train = np.array(states)
    actions = np.array(actions)

    print("shape of x_test: ", X_test.shape)

    examples_test = []
    for k in range(num_baits):
        examples_test.append(X_test[np.argwhere(y_test == k).flatten()])

    actions_test = []
    states_test = []
    for ex in X_test:
        if not use_actions:
            states_test.append(ex[:, :16])
            actions_test.append(ex[:, 16:])
        else:
            states_test.append(ex)
            actions_test.append(ex[:, 16:])

    X_test = np.array(states_test)
    actions_test = np.array(actions_test)

    return X_train, y_train, actions, X_test, y_test, actions_test


def save_plots(history, use_auxiliary_pred=False, use_classifier=False):
    fig_loss, ax_loss = plt.subplots()
    fig_loss.suptitle('Training Losses')
    if use_classifier:
        ax_loss.plot(history.history['classification_loss'], label="classifier_loss")

    if use_auxiliary_pred:
        print("inside wrong")
        ax_loss.plot(history.history['loss'], label="train_total")
        ax_loss.plot(history.history['decoder_loss'], label="traj_decoder")
        ax_loss.plot(history.history['action_predictor_loss'], label="action_decoder")
        ax_loss.plot(history.history['val_loss'], label="val")

    else:
        if use_classifier:
            ax_loss.plot(history.history['decoder_loss'], label="traj_decoder")

        ax_loss.plot(history.history['loss'], label="train_total")
        ax_loss.plot(history.history['val_loss'], label="val")

    ax_loss.legend()

    return fig_loss


def load_encoder(model_test, weights_path):
    model_test.load_weights(weights_path)
    layer_name = 'encoder'
    encoder = Model(inputs=model_test.input, outputs=model_test.get_layer(layer_name).output)

    return encoder


def plot_pca_embeddings(emb, num_baits, y_train, data_path, experiment_name, test_data=False,
                        agent_key=None, suffix=''):
    fig_pca, ax_pca = plt.subplots()
    fig_pca.suptitle('PCA')
    print("agent key: ", agent_key)
    pca_emb = []
    for k in range(num_baits):
        pca_emb.append(emb[np.argwhere(y_train == k).flatten()])

    print("shape of pca_emb: ", np.array(pca_emb).shape)
    for j in range(0, len(pca_emb)):
        print(pca_emb[j].shape)

        if agent_key is None:
            ax_pca.scatter(pca_emb[j][:, 0], pca_emb[j][:, 1], label=str(j))
        else:
            ax_pca.scatter(pca_emb[j][:, 0], pca_emb[j][:, 1], label=agent_key[j])

        if not test_data:
            np.save(data_path + experiment_name + '/pca' + suffix + '_class' + str(j) + '.npy', pca_emb[j])
        else:
            np.save(data_path + experiment_name + '/pca_test' + suffix + '_class' + str(j) + '.npy', pca_emb[j])

    ax_pca.legend()
    return fig_pca, ax_pca


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


def getStateArray(state_dict, bait_id=0, shells=True):
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


def trajectory_from_json(log):
    ep_states, ep_actions = [], []

    for state in log['game_states']:
        json_data = getStateDict_Json(state)
        states, actions = getStateArray(json_data)
        ep_states.append(states)
        ep_actions.append(actions)

    ep_states = np.array(ep_states)
    ep_actions = np.array(ep_actions)

    return ep_states, ep_actions


def get_agent_trajectory(episode_states, episode_actions, agent_id=1):
    assert agent_id == 0 or agent_id == 1
    if agent_id == 0:
        agent_states = np.concatenate((episode_states[:, :10], episode_states[:,  16:19],
                                       episode_states[:, 22:25]), axis=1)
        agent_actions = episode_actions[:, :3]

    elif agent_id == 1:
        agent_states = np.concatenate((episode_states[:, :4], episode_states[:, 10:16],
                                       episode_states[:, 19:22], episode_states[:, 25:]), axis=1)
        agent_actions = episode_actions[:, 3:]

    return agent_states, agent_actions


def get_classifier_opts():
    parser = argparse.ArgumentParser(description='TSF_RL: Space Fortress')

    # Paths

    parser.add_argument('--agents',  nargs='+', type=str,
                        help='bait paths')

    parser.add_argument('--state_sequences_only', action="store_true", default=False,
                        help='whether to train only on state sequences')

    parser.add_argument('--rollout_path', type=str, default='/serverdata/sid/self_play_trajectories/',
                        help='path to bait policies')

    parser.add_argument('--data_path', type=str, default='data/',
                        help='directory to store classification model')

    parser.add_argument('--plots_path', type=str, default='plots/',
                        help='directory to store classification model')

    parser.add_argument('--experiment_name', type=str, default='classification_baits',
                        help='Name of the experiment')

    parser.add_argument('--generate_data', action="store_true", default=False,
                        help='whether to generate or load evaluate')

    parser.add_argument('--load_baits', action="store_true", default=False,
                        help='whether to load baits or shooters')

    parser.add_argument('--num_agents', type=int, default=2,
                        help='number of baits')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of baits')

    parser.add_argument('--train', action="store_true", default=False,
                        help='whether to train or not.')

    opts = parser.parse_args()

    return opts


def train_generator_classifier(X_train, y_train, num_agents):
    while True:
        # sequence_length = np.random.randint(10, 100)
        ind = np.random.randint(len(X_train))
        x_gen = X_train[ind].reshape((1, X_train[ind].shape[0], X_train[ind].shape[1]))

        y = np.zeros((X_train[ind].shape[0], num_agents))
        print(y_train[ind])
        y[:, y_train[ind]] = 1.0
        y = y.reshape((1, y.shape[0], y.shape[1]))

        yield x_gen, y


def val_generator_classifier(X_test, y_test, num_agents):
    while True:
        # sequence_length = np.random.randint(10, 100)
        ind = np.random.randint(len(X_test))

        x_gen = X_test[ind].reshape((1, X_test[ind].shape[0], X_test[ind].shape[1]))
        # y_train will depend on past 5 timesteps of x

        y = np.zeros((X_test[ind].shape[0], num_agents))
        y[:, y_test[ind]] = 1.0

        y = y.reshape((1, y.shape[0], y.shape[1]))
        yield x_gen, y


def get_classifier_model(num_baits, train_states_only=False):
    model = Sequential()

    if not train_states_only:
        print("in the right loop")
        model.add(Bidirectional(LSTM(8, return_sequences=True), input_shape=(None, 19)))
        # model.add(Bidirectional(LSTM(8), input_shape=(None, 19)))

    else:
        model.add(LSTM(4, input_shape=(None, 16), return_sequences=True))

    model.add(Dropout(0.5))
    # model.add(Bidirectional(LSTM(4, return_sequences=True)))
    # model.add(Dropout(0.5))

    # model.add(TimeDistributed(Dense(num_baits, activation='softmax')))
    model.add(Dense(num_baits, activation='softmax'))

    optimizer = Adam(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    print(model.summary())

    return model


def plot_json_embeddings(opts, embeddings):
    """
    plots the pca of the given embeddings along with the embeddings of the training and validation data.

    """
    suffix = ''
    if opts.use_auxiliary_pred:
        suffix += '_aux_pred'

    if opts.use_actions:
        suffix += '_use_actions'

    if opts.use_classifier:
        suffix += '_use_classifier'

    data_path = opts.data_path + opts.experiment_name
    with open(data_path + "/agent_key.txt", "rb") as fp:
        agent_key = pickle.load(fp)

    fig, ax = plt.subplots()
    fig_test, ax_test = plt.subplots()

    pca_model_train = pickle.load(open(data_path + '/model_pca' + suffix + '.pkl', 'rb'))
    pca_model_test = pickle.load(open(data_path + '/model_pca_test' + suffix + '.pkl', 'rb'))

    pca_embeddings, pca_embeddings_test = [], []

    print("shape of embeddings: ", embeddings.shape)
    for k in range(len(embeddings)):
        pca_embeddings.append(pca_model_train.transform(embeddings[k]))
        pca_embeddings_test.append(pca_model_test.transform(embeddings[k]))

    pca_embeddings_test = np.array(pca_embeddings_test)
    pca_embeddings = np.array(pca_embeddings)

    for k in range(opts.num_agents):

        pca = np.load(data_path + '/pca' + suffix + '_class' + str(k) + '.npy')
        pca_test = np.load(data_path + '/pca_test' + suffix + '_class' + str(k) + '.npy')

        ax.scatter(pca[:, 0], pca[:, 1], label=agent_key[k])
        ax_test.scatter(pca_test[:, 0], pca_test[:, 1], label=agent_key[k])

    ax.scatter(pca_embeddings[:, :, 0], pca_embeddings[:, :,  1], label="your agent")
    ax_test.scatter(pca_embeddings_test[:, :, 0], pca_embeddings_test[:, :, 1], label="your agent")

    ax.legend()
    ax_test.legend()
    fig.savefig(opts.plots_path + opts.experiment_name + '/pca_json.png')
    fig_test.savefig(opts.plots_path + opts.experiment_name + '/pca_test_json.png')


def evaluate_performance(log, bait_id=1):

    """

    processes a json file and returns number of fortress kills and bait deaths.

    """

    ep_fortress_status, ep_bait_status = [], []

    for state in log['game_states']:

        ep_fortress_status.append(state["fortresses"][str(0)]["alive"])
        ep_bait_status.append(state["players"][str(bait_id)]["alive"])

    num_fortress_kills, num_bait_deaths = 0, 0
    for k in range(len(ep_bait_status) - 1):
        if ep_bait_status[k] and not ep_bait_status[k + 1]:
            num_bait_deaths += 1
        if ep_fortress_status[k] and not ep_fortress_status[k + 1]:
            num_fortress_kills += 1

    return num_fortress_kills, num_bait_deaths