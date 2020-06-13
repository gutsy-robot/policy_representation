from keras.layers import Input
from keras.models import Model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Lambda
from keras.layers import RepeatVector
from keras.layers import TimeDistributed, Dropout
from keras.callbacks import ModelCheckpoint
import numpy as np
from keras import backend as K
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
import pickle


def get_opts():
    parser = argparse.ArgumentParser(description='TSF_RL: Space Fortress')
    parser.add_argument('--use_auxiliary_pred', type=str, default="True", help='use auxiliary predictions')
    parser.add_argument('--experiment_name', type=str, default="classification_baits", help='Experiment Name')
    parser.add_argument('--use_actions', type=str, default="False", help='use actions')
    parser.add_argument('--use_classifier', type=str, default="False", help='use classifier')
    parser.add_argument('--num_agents', type=int, default=2, help='Number of baits')
    parser.add_argument('--epochs', type=int, default=1, help='Number of baits')
    parser.add_argument('--data_path', type=str, default="policy_representation/bait_classifiers/", help='data_path')
    parser.add_argument('--train', type=str, default="True", help='train')

    opts = parser.parse_args()
    return opts


# custom layer for creating repeating encodings for the decoder to decode to appropriate length.
def repeat_vector(args):
    layer_to_repeat = args[0]
    sequence_layer = args[1]
    return RepeatVector(K.shape(sequence_layer)[1])(layer_to_repeat)


# generator to generate variable length inputs for the autoencoder.
def train_generator(X_train, y_train, actions=None, num_baits=2, use_auxiliary_pred=False, use_classifier=False):
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
                y = np.zeros((1, num_baits))
                # print(y_train[ind])
                y[:, y_train[ind]] = 1.0
                # y = y.reshape((1, y.shape[0], y.shape[1]))
                yield x_gen, {'decoder': y_gen, 'action_predictor': act, 'classification': y}

        else:
            if not use_classifier:
                yield x_gen, y_gen
            else:
                y = np.zeros((1, num_baits))
                y[:, y_train[ind]] = 1.0
                yield x_gen, {'decoder': y_gen, 'classification': y}


def val_generator(X_test, y_test, actions_test=None, num_baits=2, use_auxiliary_pred=False,
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
                y = np.zeros((1, num_baits))

                # print(y_train[ind])
                y[:, y_test[ind]] = 1.0
                # y = y.reshape((1, y.shape[0], y.shape[1]))
                yield x_gen, {'decoder': y_gen, 'action_predictor': act, 'classification': y}

        else:
            if not use_classifier:
                yield x_gen, y_gen

            else:
                y = np.zeros((1, num_baits))
                y[:, y_test[ind]] = 1.0
                yield x_gen, {'decoder': y_gen, 'classification': y}


def get_model(use_auxiliary_pred=False, use_actions=False, use_classifier=False, num_baits=2):
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
        softmax_out = Dense(num_baits, activation='softmax', name='classification')(output_1)

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
            model = Model(inputs=inp, outputs=predictions)
            model.compile(optimizer='adam', loss='mse')
        else:
            model = Model(inputs=inp, outputs=[predictions, softmax_out])
            model.compile(optimizer='adam', loss={'decoder': 'mean_squared_error',
                                                  'classification': 'categorical_crossentropy'})
    return model


def train(model, X_train, y_train, actions, X_test, y_test, actions_test, data_path, experiment_name, num_baits=2,
          use_classifier=False, use_auxiliary_pred=False, epochs=1):
    checkpoint = ModelCheckpoint(data_path + experiment_name + '/model_ae.hdf5',
                                 monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='min')
    history = model.fit_generator(train_generator(X_train, y_train, actions=actions,
                                                  num_baits=num_baits, use_classifier=use_classifier,
                                                  use_auxiliary_pred=use_auxiliary_pred),
                                  validation_data=val_generator(X_test, y_test, num_baits=num_baits,
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
        ax_loss.plot(history.history['loss'], label="total")
        ax_loss.plot(history.history['decoder_loss'], label="state_decoder")
        ax_loss.plot(history.history['action_predictor_loss'], label="action_decoder")
        ax_loss.plot(history.history['val_loss'], label="val")

    else:
        print("right plotting block")
        ax_loss.plot(history.history['loss'], label="train")
        ax_loss.plot(history.history['val_loss'], label="val")

    ax_loss.legend()

    return fig_loss


def load_encoder(model_test, weights_path):
    model_test.load_weights(weights_path)
    layer_name = 'encoder'
    encoder = Model(inputs=model_test.input, outputs=model_test.get_layer(layer_name).output)

    return encoder


def plot_pca_embeddings(emb, num_baits, y_train, data_path, experiment_name):
    fig_pca, ax_pca = plt.subplots()
    fig_pca.suptitle('PCA')

    pca_emb = []
    for k in range(num_baits):
        pca_emb.append(emb[np.argwhere(y_train == k).flatten()])

    print("shape of pca_emb: ", np.array(pca_emb).shape)
    for j in range(0, len(pca_emb)):
        print(pca_emb[j].shape)

        ax_pca.scatter(pca_emb[j][:, 0], pca_emb[j][:, 1], label=str(j))
        np.save(data_path + experiment_name + '/pca_class' + str(j) + '.npy', pca_emb[j])

    ax_pca.legend()
    return fig_pca

