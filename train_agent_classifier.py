import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, TimeDistributed, Dropout, Bidirectional
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import argparse, os
from keras.optimizers import Adam


def load_data(bait_ids, baits_path, train_on_states_only=False, load_baits=True):
    """
    bait_ids: name of bait/shooter policies
    train_on_states_only: Whether to train on states only or both states and actions
    baits_path: Common Directory for bait trajectories

    """
    num_baits = len(bait_ids)
    train_x = []
    train_y = []

    # for figuring how much zero padding needs to be added.
    max_steps = 0

    # store actual length of each episode to get back the actual length sequences after scaling.
    episode_lengths = []

    for i in range(0, len(bait_ids)):
        # load training data
        bait = bait_ids[i]
        if not load_baits:
            actions_path = baits_path + bait + '/shooter_actions.npy'
            state_path = baits_path + bait + '/shooter_states.npy'
            actions = np.load(actions_path, allow_pickle=True)
            states = np.load(state_path, allow_pickle=True)

        else:
            actions_path = baits_path + bait + '/bait_actions.npy'
            state_path = baits_path + bait + '/bait_states.npy'
            actions = np.load(actions_path, allow_pickle=True)
            states = np.load(state_path, allow_pickle=True)

        for j in range(0, len(actions)):
            episode_states = np.array(states[j])
            episode_actions = np.array(actions[j])
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

    print("max steps are: ", max_steps)
    t = []

    # add zero padding before scaling.
    for j in range(0, len(train_x)):
        x = train_x[j]
        if len(x) < max_steps:
            padding = np.zeros((max_steps - len(x), x.shape[1]))
            x = np.vstack((x, padding))
        t.append(x)

    t = np.array(t)
    train_x = t
    print("shape of train_x is: ", train_x.shape)
    # for t in train_x:
    #     if t.shape != (1501, 19):
    # print(t.shape)

    assert len(train_x.shape) == 3

    actual_shape = train_x.shape

    train_x = train_x.reshape((actual_shape[0], actual_shape[1] * actual_shape[2]))
    print("shape: ", train_x.shape)

    scale = np.abs(train_x).max(axis=0) + 0.000001
    train_x = train_x / (np.abs(train_x).max(axis=0) + 0.000001)

    # reshape to the actual feature.
    train_x = train_x.reshape(actual_shape)

    # make each episode of its original length.
    x = []

    for k in range(0, len(train_x)):
        episode = train_x[k]
        original_length = episode_lengths[k]

        episode_actual = episode[:original_length]
        x.append(episode_actual)

    x = np.array(x)
    train_x = x

    print("shape of train_x before train test split: ", train_x.shape)
    print("shape of train_y before train test split: ", train_y.shape)

    # shuffle data
    X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

    # np.savez(save_path + '/data.npz', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    return X_train, X_test, y_train, y_test, scale


def get_opts():
    parser = argparse.ArgumentParser(description='TSF_RL: Space Fortress')

    # Paths

    parser.add_argument('--agents', type=list,
                        default=['Mirror_Shooter0_Yikang0', 'Turn_Only_Shooter3_Yikang0'],
                        help='bait paths')

    parser.add_argument('--state_sequences_only', type=str, default="False",
                        help='whether to train only on state sequences')

    parser.add_argument('--rollout_path', type=str, default='../new/Team-Space-Fortress/hat/self_play/trajectories/',
                        help='path to bait policies')

    parser.add_argument('--model_path', type=str, default='policy_representation/bait_classifiers/',
                        help='directory to store classification model')

    parser.add_argument('--exp_name', type=str, default='classification_baits',
                        help='Name of the experiment')

    # parser.add_argument('--train', type=str, default="True",
    #                     help='whether to train or only evaluate')
    parser.add_argument('--generate_data', type=str, default="True",
                        help='whether to generate or load evaluate')

    parser.add_argument('--load_baits', type=str, default="True",
                        help='whether to load baits or shooters')

    parser.add_argument('--num_agents', type=int, default=2,
                        help='number of baits')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of baits')

    opts = parser.parse_args()

    return opts


def train_generator(X_train, y_train):
    while True:
        # sequence_length = np.random.randint(10, 100)
        ind = np.random.randint(len(X_train))
        x_gen = X_train[ind].reshape((1, X_train[ind].shape[0], X_train[ind].shape[1]))

        y = np.zeros((X_train[ind].shape[0], num_agents))
        print(y_train[ind])
        y[:, y_train[ind]] = 1.0
        y = y.reshape((1, y.shape[0], y.shape[1]))

        yield x_gen, y


def val_generator(X_test, y_test):
    while True:
        # sequence_length = np.random.randint(10, 100)
        ind = np.random.randint(len(X_test))

        x_gen = X_test[ind].reshape((1, X_test[ind].shape[0], X_test[ind].shape[1]))
        # y_train will depend on past 5 timesteps of x

        y = np.zeros((X_test[ind].shape[0], num_agents))
        y[:, y_test[ind]] = 1.0

        y = y.reshape((1, y.shape[0], y.shape[1]))
        yield x_gen, y


def get_model(num_baits, train_states_only=False):
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


opts = get_opts()
agent_ids = opts.agents
train_on_states = opts.state_sequences_only == "True"
rollout_path = opts.rollout_path
model_path = opts.model_path
exp_name = opts.exp_name
num_agents = opts.num_agents
# train = opts.train == "True"
generate = opts.generate_data == "True"
load_baits = opts.load_baits == "False"
epochs = opts.epochs

try:
    os.mkdir(model_path + exp_name)

except Exception as e:
    print(e)

if generate:
    X_train, X_test, y_train, y_test, scale = load_data(agent_ids, rollout_path, train_on_states, load_baits=load_baits)

    np.save('policy_representation/X_train_' + exp_name + '.npy', X_train)
    np.save('policy_representation/y_train_' + exp_name + '.npy', y_train)

    np.save('policy_representation/X_test_' + exp_name + '.npy', X_test)
    np.save('policy_representation/y_test_' + exp_name + '.npy', y_test)

    np.save('policy_representation/scale_' + exp_name + '.npy', scale)


else:
    X_train = np.load('policy_representation/X_train_' + exp_name + '.npy')
    X_test = np.load('policy_representation/X_test_' + exp_name + '.npy')

    y_train = np.load('policy_representation/y_train_' + exp_name + '.npy')
    y_test = np.load('policy_representation/y_test_' + exp_name + '.npy')


model = get_model(num_agents, train_states_only=train_on_states)
checkpoint = ModelCheckpoint(model_path + exp_name + '/classifier.hdf5',
                             monitor='accuracy', verbose=1, save_best_only=True, mode='max')

history = model.fit_generator(train_generator(X_train, y_train), validation_data=val_generator(X_test, y_test),
                              validation_steps=1,
                              steps_per_epoch=30, epochs=epochs, verbose=1, callbacks=[checkpoint])

model.save(model_path + exp_name + '/classifier.h5')
print(history.history.keys())
train_loss = history.history['loss']
train_acc = history.history['accuracy']

val_loss = history.history['val_loss']
val_acc = history.history['val_accuracy']

# plot data
fig, ax = plt.subplots()
ax.plot(train_loss, label="train_loss")

ax.legend()

fig_acc, ax_acc = plt.subplots()
ax_acc.plot(train_acc, label="train_acc")

ax_acc.legend()
fig_val, ax_val = plt.subplots()
ax_val.plot(val_acc, label="val_acc")
ax_val.plot(val_loss, label="val_loss")
ax_val.legend()
fig.savefig(model_path + exp_name + '/loss.png')
fig_acc.savefig(model_path + exp_name + '/acc.png')
fig_val.savefig(model_path + exp_name + '/val.png')
