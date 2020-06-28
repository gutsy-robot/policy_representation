from keras.models import Model
import numpy as np
from keras import backend as K
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
import pickle
from policy_representation.utils import get_opts, prepare_data, train, save_plots, load_encoder, get_model, \
    plot_pca_embeddings

opts = get_opts()
use_auxiliary_pred = opts.use_auxiliary_pred
experiment_name = opts.experiment_name
use_actions = opts.use_actions
use_classifier = opts.use_classifier
num_agents = opts.num_agents
data_path = opts.data_path
plots_path = opts.plots_path
is_train = opts.train
epochs = opts.epochs

suffix = ''
if use_auxiliary_pred:
    suffix += '_aux_pred'

if use_actions:
    suffix += '_use_actions'

if use_classifier:
    suffix += '_use_classifier'

# load data
X_train = np.load(data_path + experiment_name + "/X_train.npy", allow_pickle=True)
y_train = np.load(data_path + experiment_name + "/y_train.npy", allow_pickle=True)
X_test = np.load(data_path + experiment_name + "/X_test.npy", allow_pickle=True)
y_test = np.load(data_path + experiment_name + "/y_test.npy", allow_pickle=True)
with open(data_path + experiment_name + "/agent_key.txt", "rb") as fp:
    agent_key = pickle.load(fp)

print(set(y_train))
# prepare data
X_train, y_train, actions, X_test, y_test, actions_test = prepare_data(X_train, y_train, X_test, y_test, num_agents,
                                                                       use_actions=use_actions)

# train model
if is_train:
    model = get_model(use_auxiliary_pred=use_auxiliary_pred, use_actions=use_actions, use_classifier=use_classifier,
                      num_agents=num_agents)
    print(model.summary())
    history = train(model, X_train, y_train, actions, X_test, y_test, actions_test, data_path=data_path,
                    experiment_name=experiment_name, use_classifier=use_classifier, num_agents=num_agents,
                    use_auxiliary_pred=use_auxiliary_pred, epochs=epochs, suffix=suffix)
    fig_loss = save_plots(history, use_auxiliary_pred=use_auxiliary_pred, use_classifier=use_classifier)
    fig_loss.savefig(plots_path + experiment_name + '/loss' + suffix + '.png')

# for getting output of the encoder.
model_test = get_model(num_agents=num_agents, use_auxiliary_pred=use_auxiliary_pred, use_actions=use_actions,
                       use_classifier=use_classifier)
encoder = load_encoder(model_test, data_path + experiment_name + '/model_ae' + suffix + '.hdf5')

encoder.save(data_path + experiment_name + '/encoder' + suffix + '.h5')

a = encoder.predict(X_train)
a_test = encoder.predict(X_test)
print(a.shape)
# Visualisation using the PCA.
pca = PCA(n_components=2)
pca_test = PCA(n_components=2)
emb = pca.fit_transform(a)
emb_test = pca_test.fit_transform(a_test)
print("shape of emb is: ", emb.shape)
pickle.dump(pca, open(data_path + experiment_name + '/model_pca' + suffix + '.pkl', "wb"))
pickle.dump(pca_test, open(data_path + experiment_name + '/model_pca_test' + suffix + '.pkl', "wb"))

fig_pca, _ = plot_pca_embeddings(emb, num_agents, y_train, data_path, experiment_name, agent_key=agent_key,
                                 suffix=suffix)
fig_pca.savefig(plots_path + experiment_name + '/pca' + suffix + '.png')

fig_pca_test, _ = plot_pca_embeddings(emb_test, num_agents, y_test, data_path, experiment_name,
                                      test_data=True, agent_key=agent_key, suffix=suffix)
fig_pca_test.savefig(plots_path + experiment_name + '/pca_test' + suffix + '.png')

# not very essential
if use_classifier:
    encoded_classifier = Model(inputs=model_test.input, outputs=model_test.get_layer('classification').output)
    encodings_classified = encoded_classifier.predict(X_train)
    labels = np.argmax(encodings_classified, axis=1)
    assert len(labels) == len(y_train)
    print("labels: ", set(list(labels)))
    print("accuracy is: ", np.sum(labels == y_train) / len(y_train))

# encoded_x = None
# encoded_y = np.zeros(a.shape[0])
# for k in range(num_agents):
#     if encoded_x is None:
#         encoded_x = a[np.argwhere(y_train == k).flatten()]
#     else:
#         encoded_x = np.concatenate((encoded_x, a[np.argwhere(y_train == k).flatten()]))
#     # encoded_y[np.argwhere(y_train == k).flatten()][k] = 1.0
#     encoded_y[np.argwhere(y_train == k).flatten()] = k
#
# encoded_x, encoded_y = shuffle(encoded_x, encoded_y)
# clf = LogisticRegression(random_state=0).fit(encoded_x, encoded_y)
# print("classification score on encoded stuff: ", clf.score(encoded_x, encoded_y))
#
# # code for evaluating
# encoded_pca_x = None
# encoded_pca_y = np.zeros(emb.shape[0])
#
# for k in range(num_agents):
#     if encoded_pca_x is None:
#         encoded_pca_x = emb[np.argwhere(y_train == k).flatten()]
#     else:
#         encoded_pca_x = np.concatenate((encoded_pca_x, emb[np.argwhere(y_train == k).flatten()]))
#
#     encoded_pca_y[np.argwhere(y_train == k).flatten()] = k
# encoded_pca_x, encoded_pca_y = shuffle(encoded_pca_x, encoded_pca_y)
# clf = LogisticRegression(random_state=0).fit(encoded_pca_x, encoded_pca_y)
# print("classification score on pca stuff: ", clf.score(encoded_pca_x, encoded_pca_y))