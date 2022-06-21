import pickle
import numpy as np

with open('dict_xTrue_dict1.pk', 'rb') as f:
    dict1 = pickle.load(f)

dict_new = {}
for pos in dict1.keys():
    xyz = (pos[0], pos[1], pos[2])
    dict_new[xyz]={}
    for lmid in dict1[pos].keys():
        if len(dict1[pos][lmid])!=0:
            dict_new[xyz][lmid]=np.median(dict1[pos][lmid])

with open('dict1_median.pk', 'wb') as f:
    print("dict1_median: ", dict_new)
    pickle.dump(dict_new, f)

# print("dict new is: ", dict_new)