from numpy import array
import pickle as pk

with open('agents/policy_representation/baits_embedding_dict.pickle','rb') as fp:
    bait_embedding_list = pk.load(fp)
with open('agents/policy_representation/shooters_embedding_dict.pickle','rb') as fp:
    shooter_embedding_list = pk.load(fp)