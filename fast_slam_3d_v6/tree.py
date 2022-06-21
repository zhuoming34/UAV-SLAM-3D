import networkx as nx
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from networkx.algorithms import approximation as approx
import random
import numpy as np
import copy
import pickle

def round_nodes(list):

    temp = []

    for item in list:

        temp.append((round(item[0]), round(item[1])))

    return temp

if __name__ == '__main__':

    G = nx.grid_2d_graph(41, 41)

    fixed_positions = {}
    for item in G.nodes:
        fixed_positions[item] = item

    fixed_nodes = fixed_positions.keys()

    pos = nx.spring_layout(G, pos=fixed_positions, fixed=fixed_nodes)

    nx.draw_networkx_nodes(G, pos, node_size=50, node_color='lightgray')

    # nx.draw_networkx_edges(G, pos, arrows=False, edge_color='g')

    target_real = round_nodes([(1, 12, 1.5), (11, 7, 2.5), (40, 0, 0), (31, 14, 0.5), (20, 20, 3), (6, 25, 0.75),
                   (40, 25, 1), (27, 30, 0.25), (2, 39, 2), (23, 40, 1.75)])

    # print("target real 0", target_real[0])

    target0 = round_nodes([(0.89, 12.34, 1.77), #0
                           (10.34, 6.58, 1.39),  # 1
                           (39.11, 0, 0),  # 2
                           (30.63, 13.71, 3.35),  # 3
                           (19.54, 19.07, 1.28),  # 4
                           (5.33, 24.97, 0.02),  # 5
                           (40, 23.76, 0),  # 6
                           (26.43, 29.14, 0),  # 7
                           (1.84, 38.17, 0.55),  # 8
                           (23.00, 39.19, 0.69),  # 9
                           ] #0
    )
    print("target0", target0)
    tree0 = [(0, 0), (1, 12), (10, 7), (39, 0), (31, 14), (20, 19), (5, 25), (40, 24), (26, 29), (2, 38), (23, 39)]
    road0 = []





    target1 = round_nodes([(0, 0), #0 ++
                           (21.60, 2.54),  # 1 ++
                           (39.81, 0),  # 2
                           (15.02, 9.75),  # 3 ++
                           (20.18, 19.46),  # 4
                           (4.80, 25.49),  # 5
                           (29.76, 24.65),  # 6
                           (25.03, 34.44),  # 7
                           (0, 40.27),  # 8
                           (40.08, 39.91),  # 9
                           ] #1
    )
    print("target1", target1)
    tree1 = [(16, 11), (), (40, 0), (), (20, 19), (5, 25), (30, 25), (25, 34), (0, 40), (40, 40)]
    road1 = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (10, 0), (11, 0), (12, 0), (13, 0), (14, 0), (15, 0), (16, 0), (17, 0), (18, 0), (19, 0), (20, 0), (21, 0), (22, 0), (22, 1), (22, 2), (22, 3), (22, 4), (23, 3), (23, 4), (23, 5), (22, 5), (21, 5), (21, 4), (21, 3), (22, 6), (22, 7), (22, 8), (22, 9), (22, 10), (22, 11), (21, 11), (20, 11), (19, 11), (18, 11), (17, 11), (16, 11), (16, 10), (17, 10), (17, 12), (16, 12), (15, 12), (15, 11), (15, 10)]

#
    target2 = round_nodes([(0, 0), #0 ++
                           (21.60, 2.54),  # 1 ++
                           (39.83, 0),  # 2 ++
                           (15.02, 9.75),  # 3 ++
                           (20.19, 19.46),  # 4
                           (4.80, 25.49),  # 5
                           (29.76, 24.65),  # 6
                           (25.03, 34.44),  # 7
                           (0, 40.27),  # 8
                           (40.08, 39.91),  # 9
                           ] #2
                          )#
    # print("target2", target2)
    tree2 = [(40, 0), (), (), (), (20, 19), (5, 25), (30, 25), (25, 34), (0, 40), (40, 40)]
    road2 = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (10, 0), (11, 0), (12, 0), (13, 0), (14, 0), (15, 0), (16, 0), (17, 0), (18, 0), (19, 0), (20, 0), (21, 0), (22, 0), (22, 1), (22, 2), (22, 3), (22, 4), (23, 3), (23, 4), (23, 5), (22, 5), (21, 5), (21, 4), (21, 3), (22, 6), (22, 7), (22, 8), (22, 9), (22, 10), (22, 11), (21, 11), (20, 11), (19, 11), (18, 11), (17, 11), (16, 11), (16, 10), (17, 10), (17, 12), (16, 12), (15, 12), (15, 11), (15, 10), (23, 0), (24, 0), (25, 0), (26, 0), (27, 0), (28, 0), (29, 0), (30, 0), (31, 0), (32, 0), (33, 0), (34, 0), (35, 0), (36, 0), (37, 0), (38, 0), (39, 0), (40, 0), (40, 1), (39, 1)]

    target3 = round_nodes([(0, 0),  # 0 ++
                           (21.60, 2.54),  # 1 ++
                           (39.83, 0),  # 2 ++
                           (15.02, 9.75),  # 3 ++
                           (19.18, 19.80),  # 4 ++
                           (4.80, 25.49),  # 5
                           (29.08, 24.49),  # 6 ++
                           (23.98, 34.64),  # 7 ++
                           (0, 40.27),  # 8
                           (40.08, 39.91),  # 9
                           ]  #3
                          )  #
    print("target3", target3)
    tree3 = [(25, 34), (), (), (), (), (5, 25), (), (), (0, 40), (40, 40)]
    road3 = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (10, 0), (11, 0), (12, 0), (13, 0), (14, 0), (15, 0), (16, 0), (17, 0), (18, 0), (19, 0), (20, 0), (21, 0), (22, 0), (22, 1), (22, 2), (22, 3), (22, 4), (23, 3), (23, 4), (23, 5), (22, 5), (21, 5), (21, 4), (21, 3), (22, 6), (22, 7), (22, 8), (22, 9), (22, 10), (22, 11), (21, 11), (20, 11), (19, 11), (18, 11), (17, 11), (16, 11), (16, 10), (17, 10), (17, 12), (16, 12), (15, 12), (15, 11), (15, 10), (23, 0), (24, 0), (25, 0), (26, 0), (27, 0), (28, 0), (29, 0), (30, 0), (31, 0), (32, 0), (33, 0), (34, 0), (35, 0), (36, 0), (37, 0), (38, 0), (39, 0), (40, 0), (40, 1), (39, 1), (20, 12), (20, 13), (20, 14), (20, 15), (20, 16), (20, 17), (20, 18), (20, 19), (21, 18), (21, 19), (21, 20), (20, 20), (19, 20), (19, 19), (19, 18), (22, 19), (23, 19), (24, 19), (25, 19), (26, 19), (27, 19), (28, 19), (29, 19), (30, 19), (30, 20), (30, 21), (30, 22), (30, 23), (30, 24), (30, 25), (31, 24), (31, 25), (31, 26), (30, 26), (29, 26), (29, 25), (29, 24), (30, 27), (30, 28), (30, 29), (30, 30), (30, 31), (30, 32), (30, 33), (30, 34), (29, 34), (28, 34), (27, 34), (26, 34), (25, 34), (25, 33), (26, 33), (26, 35), (25, 35), (24, 35), (24, 34), (24, 33)]\

    target4 = round_nodes([(0, 0),  # 0 ++
                           (21.60, 2.54),  # 1 ++
                           (39.83, 0),  # 2 ++
                           (15.02, 9.75),  # 3 ++
                           (19.18, 19.80),  # 4 ++
                           (4.80, 25.49),  # 5
                           (29.08, 24.49),  # 6 ++
                           (23.98, 34.64),  # 7 ++
                           (0, 40.27),  # 8
                           (38.78, 39.73),  # 9 ++
                           ]  # 4
                          )  #
    print("target4", target4)
    tree4 = [(40, 40), (), (), (), (), (5, 25), (), (), (0, 40), ()]
    road4 = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (10, 0), (11, 0), (12, 0), (13, 0), (14, 0), (15, 0), (16, 0), (17, 0), (18, 0), (19, 0), (20, 0), (21, 0), (22, 0), (22, 1), (22, 2), (22, 3), (22, 4), (23, 3), (23, 4), (23, 5), (22, 5), (21, 5), (21, 4), (21, 3), (22, 6), (22, 7), (22, 8), (22, 9), (22, 10), (22, 11), (21, 11), (20, 11), (19, 11), (18, 11), (17, 11), (16, 11), (16, 10), (17, 10), (17, 12), (16, 12), (15, 12), (15, 11), (15, 10), (23, 0), (24, 0), (25, 0), (26, 0), (27, 0), (28, 0), (29, 0), (30, 0), (31, 0), (32, 0), (33, 0), (34, 0), (35, 0), (36, 0), (37, 0), (38, 0), (39, 0), (40, 0), (40, 1), (39, 1), (20, 12), (20, 13), (20, 14), (20, 15), (20, 16), (20, 17), (20, 18), (20, 19), (21, 18), (21, 19), (21, 20), (20, 20), (19, 20), (19, 19), (19, 18), (22, 19), (23, 19), (24, 19), (25, 19), (26, 19), (27, 19), (28, 19), (29, 19), (30, 19), (30, 20), (30, 21), (30, 22), (30, 23), (30, 24), (30, 25), (31, 24), (31, 25), (31, 26), (30, 26), (29, 26), (29, 25), (29, 24), (30, 27), (30, 28), (30, 29), (30, 30), (30, 31), (30, 32), (30, 33), (30, 34), (29, 34), (28, 34), (27, 34), (26, 34), (25, 34), (25, 33), (26, 33), (26, 35), (25, 35), (24, 35), (24, 34), (24, 33), (31, 34), (32, 34), (33, 34), (34, 34), (35, 34), (36, 34), (37, 34), (38, 34), (39, 34), (40, 34), (40, 35), (40, 36), (40, 37), (40, 38), (40, 39), (40, 40), (39, 39), (39, 40)]

    target5 = round_nodes([(0, 0),  # 0 ++
                           (21.60, 2.54),  # 1 ++
                           (39.83, 0),  # 2 ++
                           (15.02, 9.75),  # 3 ++
                           (19.18, 19.80),  # 4 ++
                           (3.97, 24.61),  # 5 ++
                           (29.08, 24.49),  # 6 ++
                           (23.98, 34.64),  # 7 ++
                           (0, 39.93),  # 8 ++
                           (38.78, 39.73),  # 9 ++
                           ]  # 5
                          )  #
    print("target5", target5)
    tree5 = [(0, 40), (), (), (), (), (), (), (), (), ()]
    road5 = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (10, 0), (11, 0), (12, 0), (13, 0), (14, 0), (15, 0), (16, 0), (17, 0), (18, 0), (19, 0), (20, 0), (21, 0), (22, 0), (22, 1), (22, 2), (22, 3), (22, 4), (23, 3), (23, 4), (23, 5), (22, 5), (21, 5), (21, 4), (21, 3), (22, 6), (22, 7), (22, 8), (22, 9), (22, 10), (22, 11), (21, 11), (20, 11), (19, 11), (18, 11), (17, 11), (16, 11), (16, 10), (17, 10), (17, 12), (16, 12), (15, 12), (15, 11), (15, 10), (23, 0), (24, 0), (25, 0), (26, 0), (27, 0), (28, 0), (29, 0), (30, 0), (31, 0), (32, 0), (33, 0), (34, 0), (35, 0), (36, 0), (37, 0), (38, 0), (39, 0), (40, 0), (40, 1), (39, 1), (20, 12), (20, 13), (20, 14), (20, 15), (20, 16), (20, 17), (20, 18), (20, 19), (21, 18), (21, 19), (21, 20), (20, 20), (19, 20), (19, 19), (19, 18), (22, 19), (23, 19), (24, 19), (25, 19), (26, 19), (27, 19), (28, 19), (29, 19), (30, 19), (30, 20), (30, 21), (30, 22), (30, 23), (30, 24), (30, 25), (31, 24), (31, 25), (31, 26), (30, 26), (29, 26), (29, 25), (29, 24), (30, 27), (30, 28), (30, 29), (30, 30), (30, 31), (30, 32), (30, 33), (30, 34), (29, 34), (28, 34), (27, 34), (26, 34), (25, 34), (25, 33), (26, 33), (26, 35), (25, 35), (24, 35), (24, 34), (24, 33), (31, 34), (32, 34), (33, 34), (34, 34), (35, 34), (36, 34), (37, 34), (38, 34), (39, 34), (40, 34), (40, 35), (40, 36), (40, 37), (40, 38), (40, 39), (40, 40), (39, 39), (39, 40), (19, 21), (19, 22), (19, 23), (19, 24), (19, 25), (18, 25), (17, 25), (16, 25), (15, 25), (14, 25), (13, 25), (12, 25), (11, 25), (10, 25), (9, 25), (8, 25), (7, 25), (6, 25), (5, 25), (5, 24), (4, 24), (4, 25), (5, 26), (5, 27), (5, 28), (5, 29), (5, 30), (5, 31), (5, 32), (5, 33), (5, 34), (5, 35), (5, 36), (5, 37), (5, 38), (5, 39), (5, 40), (4, 40), (3, 40), (2, 40), (1, 40), (0, 40), (0, 39), (1, 39)]

    # --------------------------- plot ----------------------------------------------
    pos = nx.spring_layout(G, pos=fixed_positions, fixed=fixed_nodes)

    old_road = road0

    tree = tree0 + old_road

    G_1st = approx.steinertree.steiner_tree(G, tree)
    # G_1st = approx.steinertree.steiner_tree(G, target_6th + road_1st + road_2nd + road_3rd + road_4th + road_5th + road_6th)

    # nx.draw_networkx_nodes(G_1st, pos, node_color='peru', node_size=50)
    nx.draw_networkx_nodes(G_1st, pos, node_color='darkgoldenrod', node_size=50) # tree using darkgoldenrod
    # nx.draw_networkx_nodes(G_1st, pos, node_color='rosybrown', node_size=50) # anchors using rosybrown

    # old road
    G_node_old = nx.Graph()
    # G_node_old.add_nodes_from(naive_road)
    G_node_old.add_nodes_from(old_road)
    nx.draw_networkx_nodes(G_node_old, pos, node_color='rosybrown', node_size=50) # anchors using rosybrown

    # G_33 = nx.Graph()
    #
    # G33.add_nodes_from(target0)
    #
    # nx.draw_networkx_nodes(G33, pos, node_size=50)

    # Start
    G_node_s = nx.Graph()
    G_node_s.add_nodes_from([tree[0]])
    nx.draw_networkx_nodes(G_node_s, pos, node_color='r', node_size=80, node_shape='^')

    G_node000 = nx.Graph()
    G_node000.add_nodes_from([tree[1]])
    nx.draw_networkx_nodes(G_node000, pos, node_color='tomato', node_size=80, node_shape='d')

    G_node111 = nx.Graph()
    G_node111.add_nodes_from([tree[2]])
    nx.draw_networkx_nodes(G_node111, pos, node_color='gold', node_size=80, node_shape='d')

    G_node222 = nx.Graph()
    G_node222.add_nodes_from([tree[3]])
    nx.draw_networkx_nodes(G_node222, pos, node_color='darkorange', node_size=80, node_shape='d')

    G_node333 = nx.Graph()
    G_node333.add_nodes_from([tree[4]])
    nx.draw_networkx_nodes(G_node333, pos, node_color='lawngreen', node_size=80, node_shape='d')

    G_node444 = nx.Graph()
    G_node444.add_nodes_from([tree[5]])
    nx.draw_networkx_nodes(G_node444, pos, node_color='b', node_size=80, node_shape='d')

    G_node555 = nx.Graph()
    G_node555.add_nodes_from([tree[6]])
    nx.draw_networkx_nodes(G_node555, pos, node_color='g', node_size=80, node_shape='d')

    G_node666 = nx.Graph()
    G_node666.add_nodes_from([tree[7]])
    nx.draw_networkx_nodes(G_node666, pos, node_color='cyan', node_size=80, node_shape='d')

    G_node777 = nx.Graph()
    G_node777.add_nodes_from([tree[8]])
    nx.draw_networkx_nodes(G_node777, pos, node_color='dodgerblue', node_size=80, node_shape='d')

    G_node888 = nx.Graph()
    G_node888.add_nodes_from([tree[9]])
    nx.draw_networkx_nodes(G_node888, pos, node_color='darkviolet', node_size=80, node_shape='d')

    G_node999 = nx.Graph()
    G_node999.add_nodes_from([tree[10]])
    nx.draw_networkx_nodes(G_node999, pos, node_color='hotpink', node_size=80, node_shape='d')

    # ------------------------ Final -------------------------------------------------
    # G_node0 = nx.Graph()
    # G_node0.add_nodes_from([(0, 0)])
    # nx.draw_networkx_nodes(G_node0, pos, node_color='r', node_size=100, node_shape='x')
    #

    G_node0 = nx.Graph()
    G_node0.add_nodes_from([(22, 3)])
    nx.draw_networkx_nodes(G_node0, pos, node_color='tomato', node_size=100, node_shape='x')

    G_node1 = nx.Graph()
    G_node1.add_nodes_from([(22, 3)])
    nx.draw_networkx_nodes(G_node1, pos, node_color='gold', node_size=100, node_shape='x')

    G_node2 = nx.Graph()
    G_node2.add_nodes_from([(40, 0)])
    nx.draw_networkx_nodes(G_node2, pos, node_color='darkorange', node_size=100, node_shape='x')

    G_node3 = nx.Graph()
    G_node3.add_nodes_from([(15, 10)])
    nx.draw_networkx_nodes(G_node3, pos, node_color='lawngreen', node_size=100, node_shape='x')

    G_node4 = nx.Graph()
    G_node4.add_nodes_from([(19, 20)])
    nx.draw_networkx_nodes(G_node4, pos, node_color='b', node_size=100, node_shape='x')

    G_node5 = nx.Graph()
    G_node5.add_nodes_from([(4, 25)])
    nx.draw_networkx_nodes(G_node5, pos, node_color='g', node_size=100, node_shape='x')

    G_node6 = nx.Graph()
    G_node6.add_nodes_from([(29, 24)])
    nx.draw_networkx_nodes(G_node6, pos, node_color='cyan', node_size=100, node_shape='x')

    G_node7 = nx.Graph()
    G_node7.add_nodes_from([(24, 35)])
    nx.draw_networkx_nodes(G_node7, pos, node_color='dodgerblue', node_size=100, node_shape='x')

    G_node8 = nx.Graph()
    G_node8.add_nodes_from([(0, 40)])
    nx.draw_networkx_nodes(G_node8, pos, node_color='darkviolet', node_size=100, node_shape='x')
    #
    G_node9 = nx.Graph()
    G_node9.add_nodes_from([(39, 40)])
    nx.draw_networkx_nodes(G_node9, pos, node_color='hotpink', node_size=100, node_shape='x')

    # ------------------------ Real -------------------------------------------------
    pos = nx.spring_layout(G, pos=fixed_positions, fixed=fixed_nodes)

    G_node0 = nx.Graph()
    G_node0.add_nodes_from([target_real[0]])
    nx.draw_networkx_nodes(G_node0, pos, node_color='tomato', node_size=100, node_shape='*')

    G_node1 = nx.Graph()
    G_node1.add_nodes_from([target_real[1]])
    nx.draw_networkx_nodes(G_node1, pos, node_color='gold', node_size=100, node_shape='*')

    G_node2 = nx.Graph()
    G_node2.add_nodes_from([target_real[2]])
    nx.draw_networkx_nodes(G_node2, pos, node_color='darkorange', node_size=100, node_shape='*')

    G_node3 = nx.Graph()
    G_node3.add_nodes_from([target_real[3]])
    nx.draw_networkx_nodes(G_node3, pos, node_color='lawngreen', node_size=100, node_shape='*')

    G_node4 = nx.Graph()
    G_node4.add_nodes_from([target_real[4]])
    nx.draw_networkx_nodes(G_node4, pos, node_color='b', node_size=100, node_shape='*')

    G_node5 = nx.Graph()
    G_node5.add_nodes_from([target_real[5]])
    nx.draw_networkx_nodes(G_node5, pos, node_color='g', node_size=100, node_shape='*')

    G_node6 = nx.Graph()
    G_node6.add_nodes_from([target_real[6]])
    nx.draw_networkx_nodes(G_node6, pos, node_color='cyan', node_size=100, node_shape='*')

    G_node7 = nx.Graph()
    G_node7.add_nodes_from([target_real[7]])
    nx.draw_networkx_nodes(G_node7, pos, node_color='dodgerblue', node_size=100, node_shape='*')

    G_node8 = nx.Graph()
    G_node8.add_nodes_from([target_real[8]])
    nx.draw_networkx_nodes(G_node8, pos, node_color='darkviolet', node_size=100, node_shape='*')

    G_node9 = nx.Graph()
    G_node9.add_nodes_from([target_real[9]])
    nx.draw_networkx_nodes(G_node9, pos, node_color='hotpink', node_size=100, node_shape='*')


    plt.show()

