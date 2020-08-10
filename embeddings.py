from numpy import array
import pickle as pk

with open('baits_embedding_dict.pickle','rb') as fp:
    bait_embedding_list = pk.load(fp)
with open('shooters_embedding_dict.pickle','rb') as fp:
    shooter_embedding_list = pk.load(fp)