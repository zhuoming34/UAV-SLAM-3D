#from sklearn.cluster import KMeans
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

A_0 = -47.29
N = 1.68
rssi_b0_threshold = -58
median_threshold = 0  # init threshold for getting more points from strongest assignment
lmid = 2
target_list = [(1, 12, 1.5), (11, 7, 2.5), (40, 0, 0), (31, 14, 0.5), (20, 20, 3), (6, 25, 0.75), (40, 25, 1),
               (27, 30, 0.25), (2, 39, 2), (23, 40, 1.75)]
square_center = [(x, y) for y in range(5, 40, 10) for x in range(5, 40, 10)]
# print(square_center)
layer_grid_weights = {0: {1.0: [0, 4, 20, 24], 0.5: [1, 5, 21, 25]},
                      1: {1.0: [1, 5, 24, 28], 0.5: [0, 4, 25, 29, 6, 2]},
                      2: {1.0: [2, 6, 28, 32], 0.5: [1, 5, 29, 33, 7, 3]},
                      3: {1.0: [3, 7, 32, 36], 0.5: [2, 6, 33, 37]},
                      4: {1.0: [4, 8, 21, 25], 0.5: [22, 26, 9, 5, 24, 20]},
                      5: {1.0: [5, 9, 25, 29], 0.5: [26, 30, 10, 6, 28, 24, 4, 8]},
                      6: {1.0: [6, 10, 29, 33], 0.5: [30, 34, 11, 7, 32, 28, 5, 9]},
                      7: {1.0: [7, 11, 33, 37], 0.5: [38, 34, 10, 6, 32, 36]},
                      8: {1.0: [8, 12, 22, 26], 0.5: [23, 27, 13, 9, 25, 21]},
                      9: {1.0: [9, 13, 26, 30], 0.5: [27, 31, 14, 10, 29, 25, 8, 12]},
                      10: {1.0: [10, 14, 30, 34], 0.5: [31, 35, 15, 11, 33, 29, 9, 13]},
                      11: {1.0: [11, 15, 34, 38], 0.5: [39, 35, 14, 10, 33, 37]},
                      12: {1.0: [12, 16, 23, 27], 0.5: [17, 13, 26, 22]},
                      13: {1.0: [13, 17, 27, 31], 0.5: [18, 14, 30, 26, 12, 16]},
                      14: {1.0: [14, 18, 31, 35], 0.5: [19, 15, 34, 30, 13, 17]},
                      15: {1.0: [15, 19, 35, 39], 0.5: [18, 14, 34, 38]},
                      }

for key in layer_grid_weights.keys():
    for s_key in layer_grid_weights[key].keys():
        temp = [j for j in layer_grid_weights[key][s_key]]
        for i in range(40, 160, 40):
            layer_grid_weights[key][s_key] += [x+i for x in temp]

#for key in layer_grid_weights.keys():
    #print(layer_grid_weights[key][1.0])

grid_classes = {}
edge_id = -1
for z in range(0, 4):
    for y in range(0, 41, 10):
        for x in range(0, 41):
            if x % 10 == 0 and x != 40:
                edge_id += 1
                grid_classes[edge_id] = []
                if x != 0:
                    grid_classes[edge_id-1] += [(x, y, z)]
            grid_classes[edge_id] += [(x, y, z)]
    for x in range(0, 41, 10):
        for y in range(0, 41):
            if y % 10 == 0 and y != 40:
                edge_id += 1
                grid_classes[edge_id] = []
                if y != 0:
                    grid_classes[edge_id-1] += [(x, y, z)]
            grid_classes[edge_id] += [(x, y, z)]

#for key in grid_classes.keys():
    #print(key, grid_classes[key])

total_class = len(grid_classes.keys())


def find_grid_in_layer(centroid, grid_c):
    dist_to_center = []
    for g_c in grid_c:
        dist = math.sqrt((centroid[0][0]-g_c[0])**2+(centroid[0][1]-g_c[1])**2)
        dist_to_center += [dist]
    # print(dist_to_center)
    return dist_to_center.index(min(dist_to_center))


def cal_entropy(index, cluster_points):
    n = len(cluster_points)
    print(str(n) + "points in this case and center at grid" + str(index))
    # get the weight sides
    w1_sides = layer_grid_weights[index][1.0]
    w1_05_sides = layer_grid_weights[index][0.5]
    class_list = [0]*total_class
    # find the weight from the sides list, search own gird first, then others
    for point in cluster_points:
        # search in 1.0 weight sides
        found = 0
        for side in w1_sides:
            if point in grid_classes[side]:
                class_list[side] += 1
                found = 1
                break
        if found == 1:
            continue

        # search in 0.5 weight sides
        for side in w1_05_sides:
            if point in grid_classes[side]:
                class_list[side] += 1
                found = 1
                break
        if found == 1:
            continue

        # search in other sides
        '''
        for side in range(0, 22):
            if side in w1_sides or side in w1_05_sides:
                continue
            if point in grid_classes[side]:
                class_list[side] += 1
                break
        '''
    print("finish finding weights for " + str(sum(class_list)) + " points")
    print(class_list)
    # calculate the
    entropy = 0
    for side in range(0, total_class):
        c_i = class_list[side]
        if c_i == 0:
            continue
        if side in w1_sides:
            entropy += c_i / n * math.log10(c_i / n)
        elif side in w1_05_sides:
            entropy += 0.5 * c_i / n * math.log10(c_i / n)
        else:
            # entropy += 0.1 * c_i / n * math.log10(c_i / n)
            entropy += 0
    entropy = abs(entropy) / math.log10(total_class)
    print("Total classes: ", total_class)
    print("The entropy is:", entropy)


with open('dict1_median.pk', 'rb') as rssi_total:
    d = pickle.load(rssi_total)

# print(d)

# load dictionary with original rssi {loc1:[rssi], loc2:[rssi], ...}
d0 = {}
for key in d.keys():
    # only assign the strongest rssi using the median
    rssi_median = []
    for id in range(0, len(target_list)):
            rssi_median += [d[key][id]]
    # print(rssi_median)
    if rssi_median.index(max(rssi_median)) == lmid:
        if rssi_median[lmid] > rssi_b0_threshold:
            d0[key] = d[key][lmid]
    # threshold in assignment to increase data points
    elif abs(max(rssi_median) - rssi_median[lmid]) < median_threshold:
        if rssi_median[lmid] > rssi_b0_threshold:
            d0[key] = d[key][lmid]
#print(d0)

data_points = list(d0.keys())
print("number of data points:", len(data_points))
print(data_points)

# make pair for k-means
X = []
for key in d0.keys():
    temp = [key[0], key[1], key[2]]
    X += [temp]
'''
clusterer = KMeans(n_clusters=1, random_state=0).fit(X)
center_X = clusterer.cluster_centers_
print(center_X)
error = ((clusterer.cluster_centers_[0][0]-target_list[lmid][0])**2+(clusterer.cluster_centers_[0][1]-target_list[lmid][1])**2
          + (clusterer.cluster_centers_[0][2]-target_list[lmid][2])**2)**0.5
print("error for B" + str(lmid)+" at " + str(rssi_b0_threshold)+" is: ", error)

X_grid_index = find_grid_in_layer(center_X, square_center)
cal_entropy(X_grid_index, data_points)
'''
'''
X = []
for key in grid_classes.keys():
    for point in grid_classes[key]:
        if point[2] == 1:
            X += [point]
'''
# print(X)
# for visualization
d0_x = []
d0_y = []
d0_z = []
for loc in X:
    d0_x += [loc[0]]
    d0_y += [loc[1]]
    d0_z += [loc[2]]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_ylim([-1, 41])
ax.set_xlim([-1, 41])
ax.set_zlim([0, 3])
#ax.scatter(d0_x, d0_y, d0_z, c='r', marker='o')
ax.scatter(d0_x, d0_y, d0_z)
#ax.scatter([x for x in range(0, 41)], [y for y in range(0, 41)], [2 for z in range(0, 41)], c='r')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('strongest B' + str(lmid) + 'at ' + str(rssi_b0_threshold) + ' with ' + str(len(X))+' RSSIs ')
plt.show()
