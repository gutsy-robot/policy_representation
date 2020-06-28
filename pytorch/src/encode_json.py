from policy_representation.utils import scale_action, scale_state, get_model, load_encoder, \
    trajectory_from_json, get_agent_trajectory, plot_json_embeddings
import argparse
import json, pickle, os
import numpy as np
from keras.models import load_model


def get_opts():
    parser = argparse.ArgumentParser(description='TSF_RL: Space Fortress')

    parser.add_argument('--use_auxiliary_pred', action="store_true", default=False, help='use auxiliary predictions')
    parser.add_argument('--use_classifier', action="store_true", default=False, help='use_classifier')

    parser.add_argument('--experiment_name', type=str, default="classification_baits", help='Experiment Name')
    parser.add_argument('--use_actions', action="store_true", default=False,
                        help='whether your pretrained model uses actions or not')
    parser.add_argument('--num_agents', type=int, default=2, help='Number of agents that were used to train the model'
                                                                  'being used')
    parser.add_argument('--agent_id', type=int, default=0, help='Either or 0 or 1 depending on which you wish '
                                                                'to find the embeddings of')
    parser.add_argument('--embedding_generation_interval', type=int,
                        default=None, help='Interval after which representation should be generated')
    parser.add_argument('--data_path', type=str, default="policy_representation/data/",
                        help='Main directory where the experiment directories are')
    parser.add_argument('--plots_path', type=str, default="policy_representation/plots/",
                        help='Main directory inside which experiment directories are in which different plots are '
                             'stored')

    parser.add_argument('--json_path', type=str, help='Path to the json file you wish to process')
    opts = parser.parse_args()
    return opts


def main(opts):
    """

    returns the embeddings for the corresponding json file.
    """

    json_path = opts.json_path
    agent_id = opts.agent_id
    model_path = opts.data_path + opts.experiment_name + '/model_ae'
    # model_path = opts.data_path + opts.experiment_name + '/model_ae.h5'

    # use_actions = model.layers[0].input_shape == 19
    embedding_interval = opts.embedding_generation_interval

    suffix = ''
    if opts.use_auxiliary_pred:
        suffix += '_aux_pred'

    if opts.use_actions:
        suffix += '_use_actions'

    if opts.use_classifier:
        suffix += '_use_classifier'

    model_path += suffix
    model_path += '.h5'
    # # load the json file.

    model = load_model(model_path)

    with open(json_path) as f:
        log = json.load(f)

    # get states and action from the json.
    states, actions = trajectory_from_json(log)
    print("shape of states is: ", states.shape)
    print("shape of actions is: ", actions.shape)
    # agent states and actions from the combined states and actions.
    agent_states, agent_actions = get_agent_trajectory(states, actions, agent_id=agent_id)
    agent_states, agent_actions = scale_state(agent_states), scale_action(agent_actions)

    if opts.use_actions:
        inp = np.hstack((agent_states, agent_actions))

    else:
        inp = agent_states

    # load model
    # model = get_model(use_auxiliary_pred=opts.use_auxiliary_pred, use_actions=opts.use_actions,
    #                   use_classifier=opts.use_classifier, num_agents=opts.num_agents)
    # model = load_encoder(model, model_path)

    embeddings = []
    if embedding_interval is not None:
        ind = embedding_interval
        while ind < len(inp):
            input = inp[:ind, :]
            input = input.reshape((1, input.shape[0], input.shape[1]))
            embeddings.append(model.predict(input))
            ind += embedding_interval

    embeddings.append(model.predict(inp.reshape((1, inp.shape[0], inp.shape[1]))))
    embeddings = np.array(embeddings)

    return embeddings


args = get_opts()
if not os.path.exists(args.data_path):
    os.makedirs(args.data_path)

embeddings = main(args)
plot_json_embeddings(args, embeddings)
