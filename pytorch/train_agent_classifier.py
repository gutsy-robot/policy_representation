import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import ModelCheckpoint
import os
from policy_representation.utils import load_data
import pickle
from policy_representation.utils import get_classifier_opts, get_classifier_model, train_generator_classifier,\
    val_generator_classifier


opts = get_classifier_opts()
agent_ids = opts.agents
print(agent_ids)
# print(opts.list)
# agent_ids = [int(item)for item in opts.list.split(',')]
#

train_on_states = opts.state_sequences_only
rollout_path = opts.rollout_path
model_path = opts.data_path
plots_path = opts.plots_path
exp_name = opts.experiment_name
num_agents = opts.num_agents
# train = opts.train == "True"
generate = opts.generate_data
load_baits = opts.load_baits
epochs = opts.epochs
train = opts.train


if not os.path.exists(model_path):
    os.makedirs(model_path)

if not os.path.exists(plots_path):
    os.makedirs(plots_path)

if not os.path.exists(model_path + exp_name):
    os.makedirs(model_path + exp_name)

if not os.path.exists(plots_path + exp_name):
    os.makedirs(plots_path + exp_name)

if generate:
    X_train, X_test, y_train, y_test = load_data(agent_ids, rollout_path, train_on_states, load_baits=load_baits)
    print("shape of X_train: ", X_train.shape)
    print("shape of y_train: ", y_train.shape)
    print("shape of X_test: ", X_test.shape)
    print("shape of y_test: ", y_test.shape)

    np.save(model_path + exp_name + '/X_train.npy', X_train)
    np.save(model_path + exp_name + '/y_train.npy', y_train)

    np.save(model_path + exp_name + '/X_test.npy', X_test)
    np.save(model_path + exp_name + '/y_test.npy', y_test)

    agent_key = []
    for k in range(len(agent_ids)):
        id = agent_ids[k]
        if load_baits:
            bait_id = id.split('Bait_')[1]
            agent_key.append('Bait_' + bait_id)

        else:
            shooter_id = id.split('_Bait_')[0]
            agent_key.append(shooter_id)

    with open(model_path + exp_name + "/agent_key.txt", "wb") as fp:
        pickle.dump(agent_key, fp)

    # np.save('policy_representation/scale_' + exp_name + '.npy', scale)


else:
    X_train = np.load(model_path + exp_name + '/X_train.npy')
    X_test = np.load(model_path + exp_name + '/X_test.npy')

    y_train = np.load(model_path + exp_name + '/y_train.npy')
    y_test = np.load(model_path + exp_name + '/y_test.npy')
    with open(model_path + exp_name + "/agent_key.txt", "rb") as fp:
        agent_key = pickle.load(fp)

if train:
    model = get_classifier_model(num_agents, train_states_only=train_on_states)
    checkpoint = ModelCheckpoint(model_path + exp_name + '/classifier.hdf5',
                                 monitor='accuracy', verbose=1, save_best_only=True, mode='max')

    history = model.fit_generator(train_generator_classifier(X_train, y_train, num_agents=num_agents),
                                  validation_data=val_generator_classifier(X_test, y_test, num_agents=num_agents),
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
    fig.savefig(plots_path + exp_name + '/loss_classifier.png')
    fig_acc.savefig(plots_path + exp_name + '/acc_classifier.png')
    fig_val.savefig(plots_path + exp_name + '/val_classifier.png')
