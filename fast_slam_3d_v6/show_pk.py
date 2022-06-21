import pickle

with open('dict_xTrue_dict1.pk', 'rb') as f:

    dict = pickle.load(f)

print("dict is: ", dict)