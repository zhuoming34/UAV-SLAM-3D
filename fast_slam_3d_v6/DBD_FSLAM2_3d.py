from numpy.linalg import norm
from math import atan2
#import scipy.stats as stats
import math #HZ
import numpy as np
import math
from numpy.random import multivariate_normal
from numpy.random import randn
import matplotlib.pyplot as plt
from collections import OrderedDict

#RSSI_N=2
#RSSI_A0=-45

# These two values are from measured data. SHOULD BE THE SAME as those in simulation code.
RSSI_N = 1.68 # alpha
RSSI_A0 = -47.29 # beta


''' calcuate the distantnce between two locations'''
def fun_dist(loc1, loc2):
    tmp = np.sqrt(np.power((loc1[0]-loc2[0]),2) + np.power((loc1[1]-loc2[1]),2))
    return tmp.item()
# 07/05/2020: added fun_dist_3d
def fun_dist_3d(loc1, loc2, loc1_z, loc2_z):
    tmp = np.sqrt(np.power((loc1[0]-loc2[0]),2) + np.power((loc1[1]-loc2[1]),2) + np.power((loc1_z-loc2_z),2))
    return tmp.item()

def get_2D_data_distance_only(distance): # distance in cm.
    in_file_name = str("/home/hgz/Downloads/50m_rssi.txt")
    dict_distance_rssi_data = {}
    dict_distance_rssi_data[distance] = []
    with open(in_file_name, "r") as in_file:
        for line in in_file:
            words = line.split()
            tmprssi = float(words[4])
            dict_distance_rssi_data[distance].append(tmprssi)

    return dict_distance_rssi_data

def get_2D_data(b_ID_start, b_ID_end):
    #----- get 2-D data ---------
    #dict_beacon_loc={}
    #dict_beacon_loc[1]=[1.  ,1.]
    #dict_beacon_loc[2]=[4.  ,1.5]
    #dict_beacon_loc[3]=[2.5 ,2.5]
    #dict_beacon_loc[4]=[3.  ,4.]
    #dict_beacon_loc[5]=[0.  ,5.]
    #dict_loc_results={}
    for b_ID in range(b_ID_start, b_ID_end+1):
        np.random.seed(200)
        dict_rssi_2D = {}
        dict_rssi_2D_mean = {}
        dict_rssi_2D_std = {}
        #true_beacon_loc = dict_beacon_loc[b_ID]

        in_file_name = str("/home/hgz/Documents/parse_example/2-d/5beaconRSSI"+str(b_ID)+".csv")

        with open(in_file_name, "r") as in_file:
            for line in in_file:
                words = line.split(',')
                tmpx = int(words[1])
                tmpy = int(words[2])
                words = words[3].split('\n')
                tmprssi = float(words[0])
                if not ((tmpx, tmpy) in dict_rssi_2D):
                    dict_rssi_2D[(tmpx, tmpy)] = []
                dict_rssi_2D[(tmpx, tmpy)].append(tmprssi)
                # loc_rcvr.append( [float(words[0])/100,float(words[1])/100 ] )
                # rssi_measured_beacon_1.append( float(words[2]) )

        #------ Process the 2-D data, so we just use the average on each drone location,
        #------  to form a sequnece of data points, to do estimation.
        #z_measured_mean_rssi=[]
        #dict_rssi_2D_sorted = sorted(dict_rssi_2D)
        for tmpkey in sorted(dict_rssi_2D):
            tmp = np.mean(dict_rssi_2D[tmpkey])
            dict_rssi_2D_mean[tmpkey] = tmp
            dict_rssi_2D_std[tmpkey] = np.std(dict_rssi_2D[tmpkey])

        return dict_rssi_2D_mean, dict_rssi_2D_std, dict_rssi_2D

def get_2D_data_new():

    list_d_raw = []
    list_rssi_raw = []

    tmp_list = list(range(50, 1050, 50))
    dist_list = [5] + tmp_list
    #print(dist_list)
    #input("pause ")
    dict_dist_rssi = {}
    dict_dist_rssi_mean = {}
    dict_dist_rssi_std  = {}

    for item in dist_list:
        dict_dist_rssi[item] = []

    for tmpkey in sorted(dict_dist_rssi.keys()):
        in_file_name = str("/home/hgz/Dropbox/Drones/Experiment_Data/test-"+str(tmpkey)+"cm.txt")
        print(in_file_name)
        count = 0
        with open(in_file_name, "r") as in_file:
            for line in in_file:
                words = line.split(' ')
                #for item in words:
                #    print(item)
                if len(words)==1:
                    break
                #print(words)
                #print(words[2])
                tmp = words[2].split(",")
                tmp = float(tmp[0])
                #print(tmp)
                #if int(tmp)== -200:
                #    print("found bad one")
                if int(tmp) != -200:
                    list_d_raw.append(float(tmpkey/100.) )
                    list_rssi_raw.append( float(tmp) )
                    #count = count + 1
                    dict_dist_rssi[tmpkey].append(float(tmp))

    for tmpkey in sorted(dict_dist_rssi.keys()):
        #print(tmpkey, dict_dist_rssi[tmpkey])
        dict_dist_rssi_mean[tmpkey] = np.mean(dict_dist_rssi[tmpkey])
        dict_dist_rssi_std[tmpkey] = np.std(dict_dist_rssi[tmpkey])

    return list_d_raw, list_rssi_raw, dict_dist_rssi, dict_dist_rssi_mean, dict_dist_rssi_std

#def generate_2D_data(b_ID_start, b_ID_end, beacon_loc, xlimit, ylimit, noise_scale):
#def generate_2D_data(beacon_loc, xstart, xlimit, ystart, ylimit, noise_scale):
def generate_2D_data(beacon_loc, area_south_west_cornet, area_north_east_corner, noise_scale):
    x = beacon_loc[0] # in meter
    y = beacon_loc[1]
    xstart = area_south_west_cornet[0]
    ystart = area_south_west_cornet[1]
    xlimit = area_north_east_corner[0]
    ylimit = area_north_east_corner[1]

    #for b_ID in range(b_ID_start, b_ID_end+1):
    np.random.seed(200)
    rssi_data = {}
    distance = {}
    #dict_rssi_2D = {}
    #dict_rssi_2D_mean = {}
    #dict_rssi_2D_std = {}

    #dx = x - xlimit / 100.0  # convert to meters
    #dy = y - ylimit / 100.0
    #max_slant_range = math.sqrt(dx ** 2 + dy ** 2)
    #print("max slant range", max_slant_range)

    step_size = 10 # cm
    for i in range(xstart, xlimit+1, step_size):
        for j in range(ystart, ylimit+1, step_size):
            dx = x - i/100.0   # convert to meters
            dy = y - j/100.0
            slant_range = math.sqrt(dx ** 2 + dy ** 2)
            if slant_range <= 0.05:
                # 5 cm. This restrication is necessary, as in practice, RSSI is always negative (seems)
                slant_range = 0.05
            distance[(i, j)] = slant_range

            # make this negative, as longer distance means more negative
            # noise = (-1) * abs(np.random.randn())
            noise = np.random.randn()
            tmp = RSSI_A0 - 10 * RSSI_N * np.log10(slant_range)
            rssi_data[(i, j)] = tmp + noise * noise_scale * slant_range # linear increase

            # Empirical data from experiments. See get_empirical_data.py file.
            tmp = RSSI_A0 - 10 * RSSI_N * np.log10(slant_range)
            #std_segment = 8.02548 * slant_range / max_slant_range
            if slant_range <= 5.0:    # if distance less
                rssi_data[(i, j)] = tmp + noise * slant_range/5.0
            else:    # if distance less
                # rssi_data[(i, j)] = tmp
                # rssi_data[(i, j)] = np.random.normal(loc=tmp, scale=std_segment, size=None)
                # rssi_data[(i, j)] = np.random.normal(loc=tmp, scale=std_segment, size=None)
                #rssi_data[(i, j)] = tmp + noise * 0.1 * slant_range  # linear increase

                rssi_data[(i, j)] = tmp + noise * 5
                # noise is calculated as a standard normal random number, with std=1, mean=0,
                # so, with 68.2% probability that noise is within [-1, 1],
                # then with the same probability that noise*5 is within [-5, 5],
                # which is added to the RSSI values (typically a negative number between -20 and -80, roughly.
        #for tmpkey in sorted(distance):
        #    print(tmpkey, "\t%.2f" % distance[tmpkey], "%.2f" % rssi_data[tmpkey])
    return rssi_data, distance


''' beacon_loc, and drone_loc are all in meters '''

def generate_RSSI(beacon_loc, drone_loc, scale_noise, std_step, std_est):

    if scale_noise < 0:
        input("error in generating RSSI")

    dx = beacon_loc[0] - drone_loc[0]
    dy = beacon_loc[1] - drone_loc[1]

    slant_range = math.sqrt(dx ** 2 + dy ** 2)

    if slant_range <= 0.01: # 1 cm.
        #slant_range = 0.0001
        slant_range = 0.01
        #slant_range = 0.005  # 04/25/2020

    tmp = RSSI_A0 - 10 * RSSI_N * np.log10(slant_range)
    #tmp = np.random.normal(loc=tmp, scale = 8.025 * scale_noise) # std=8.025 is from get_empirical_data.py.
    #rssi_data = tmp

    #if scale_noise >= 0 and scale_noise <= 1 :
    #noise = np.random.randn()

    if slant_range <= std_step:  # if distance less
        tmp_std = std_est * scale_noise * slant_range/float(std_step)
    #elif 1.0 < slant_range <= 2.0:  # if distance less
    #    rssi_data = tmp + noise * slant_range / 2.0 * scale_noise
    #elif 2.0 < slant_range <= 5.0:  # if distance less
    #    rssi_data = tmp + noise * slant_range / 5.0 * scale_noise
    #elif 5.0 < slant_range <= 8.0:  # if distance less
    #    rssi_data = tmp + noise * scale_noise * 1.0
    else:
        tmp_std = std_est * scale_noise

    rssi_data = np.random.normal(loc=tmp, scale=tmp_std)
    #    rssi_data = rssi_data * scale_noise
    #else: # the case that scale_nosie = 0
    #    input("scale noise is out of range. error in generating RSSI")
    #    #rssi_data = tmp
    return rssi_data


def generate_RSSI_3d(beacon_loc, beacon_h, drone_loc, drone_h, scale_noise, std_step, std_est):

    if scale_noise < 0:
        input("error in generating RSSI")

    dx = beacon_loc[0] - drone_loc[0]
    dy = beacon_loc[1] - drone_loc[1]
    dz = beacon_h - drone_h

    #slant_range = math.sqrt(dx ** 2 + dy ** 2)
    slant_range = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    if slant_range <= 0.01: # 1 cm.
        #slant_range = 0.0001
        slant_range = 0.01
        #slant_range = 0.005  # 04/25/2020

    tmp = RSSI_A0 - 10 * RSSI_N * np.log10(slant_range)
    #tmp = np.random.normal(loc=tmp, scale = 8.025 * scale_noise) # std=8.025 is from get_empirical_data.py.
    #rssi_data = tmp

    #if scale_noise >= 0 and scale_noise <= 1 :
    #noise = np.random.randn()

    if slant_range <= std_step:  # if distance less
        tmp_std = std_est * scale_noise * slant_range/float(std_step)
    #elif 1.0 < slant_range <= 2.0:  # if distance less
    #    rssi_data = tmp + noise * slant_range / 2.0 * scale_noise
    #elif 2.0 < slant_range <= 5.0:  # if distance less
    #    rssi_data = tmp + noise * slant_range / 5.0 * scale_noise
    #elif 5.0 < slant_range <= 8.0:  # if distance less
    #    rssi_data = tmp + noise * scale_noise * 1.0
    else:
        tmp_std = std_est * scale_noise

    rssi_data = np.random.normal(loc=tmp, scale=tmp_std)
    #    rssi_data = rssi_data * scale_noise
    #else: # the case that scale_nosie = 0
    #    input("scale noise is out of range. error in generating RSSI")
    #    #rssi_data = tmp
    return rssi_data


def getDronePath(drone_init_loc, list_turn_pts=None):

    drone_path_2=[]
    # print("list_turn_pts inside DBD: ", list_turn_pts)

    for item in list_turn_pts:
        drone_path_2.append((int(item[0]), int(item[1]), 0))

    # print("list_turn_pts")
    # print(list_turn_pts)
    # print(drone_init_loc)
    # print(drone_path_2)
    '''
    if list_turn_pts == None:

        while len(drone_path_2) < 20:
            last_index = len(drone_path_2) - 1
            #print( drone_path_2[i - 1])
            #print(drone_path_2[i - 1][0])
            #print(drone_path_2[i - 1][1])
            tmp_x = drone_path_2[last_index][0]
            tmp_y = drone_path_2[last_index][1]
            directions=[1,2,3,4]
            direction_move = np.random.choice(directions)
            if direction_move == 1:
                tmp_x = tmp_x - 50
            elif direction_move == 2:
                tmp_x = tmp_x + 50
            elif direction_move == 3:
                tmp_y = tmp_y - 50
            else:
                tmp_y = tmp_y + 50

            if tmp_x < 0:
                tmp_x = 0
            if tmp_x > 300:
                tmp_x = 300
            if tmp_y < 0:
                tmp_y = 0
            if tmp_y > 300:
                tmp_y = 300

            drone_path_2.append( (tmp_x, tmp_y, 0) ) # assume yaw is 0 degree.
            drone_path_2=list(OrderedDict.fromkeys(drone_path_2))  #??? forgot why doing this

    else:
        for item in list_turn_pts:
            drone_path_2.append( (item[0], item[1], 0) )
    '''

    drone_path=[]

    #if int(drone_init_loc[0]) != int(drone_path_2[0][0]) \
    #        or int(drone_init_loc[1]) != int(drone_path_2[0][1]) :
    #    #drone_path.append( (drone_init_loc[0]*100, drone_init_loc[1]*100, 0) )
    #    #print("adding ")
    #    drone_path.append((int(drone_init_loc[0]), int(drone_init_loc[1]), 0))
    #drone_path.append((int(drone_init_loc[0]*100), int(drone_init_loc[1]*100), 0) )
    drone_path.append((int(drone_path_2[0][0]), int(drone_path_2[0][1]), 0))
    # print("drone_path inside after copy", drone_path)
    #input("pause ...\n\n")
    for i in range(1, len(drone_path_2)):
        # print("ttt len(drone_path_2)", len(drone_path_2))
        #print(i)
        #print(drone_path_2[i - 1][0])
        #print(drone_path_2[i][0])
        #print(drone_path_2[i - 1][1])
        #print(drone_path_2[i][1])

        if drone_path_2[i][0] > drone_path_2[i-1][0] and int(drone_path_2[i][1]) == int(drone_path_2[i-1][1]):
            yaw = 0
        elif drone_path_2[i][0] < drone_path_2[i-1][0] and int(drone_path_2[i][1]) == int(drone_path_2[i-1][1]):
            yaw = 180 # degree
        elif drone_path_2[i][1] > drone_path_2[i-1][1] and int(drone_path_2[i][0]) == int(drone_path_2[i-1][0]):
            yaw = 90 # degree
        elif drone_path_2[i][1] < drone_path_2[i-1][1] and int(drone_path_2[i][0]) == int(drone_path_2[i-1][0]):
            yaw = -90 # degree
        elif drone_path_2[i][1] == drone_path_2[i-1][1] and int(drone_path_2[i][0]) == int(drone_path_2[i-1][0]):
            yaw = 0
        else:
            yaw_arc = math.atan((int(drone_path_2[i][1]) - int(drone_path_2[i - 1][1])) / (
                    int(drone_path_2[i][0]) - int(drone_path_2[i - 1][0])))
            yaw = yaw_arc * 180 / math.pi
            if drone_path_2[i][0] > drone_path_2[i - 1][0] and int(drone_path_2[i][1]) > int(drone_path_2[i - 1][1]):
                yaw = yaw
            elif drone_path_2[i][0] < drone_path_2[i - 1][0] and int(drone_path_2[i][1]) > int(drone_path_2[i - 1][1]):
                yaw = 180 + yaw  # degree
            elif drone_path_2[i][0] < drone_path_2[i - 1][0] and int(drone_path_2[i][1]) < int(drone_path_2[i - 1][1]):
                yaw = yaw - 180  # degree
            elif drone_path_2[i][0] > drone_path_2[i - 1][0] and int(drone_path_2[i][1]) < int(drone_path_2[i - 1][1]):
                yaw = yaw  # degree
            else:
                yaw = 0

        if int(drone_path_2[i][1]) == int(drone_path_2[i - 1][1]) and int(drone_path_2[i][0]) == int(drone_path_2[i - 1][0]):
            continue
        else:
            drone_path.append( (int(drone_path_2[i][0]), int(drone_path_2[i][1]), yaw) )
        # 20200704 changed this
        # drone_path.append((int(drone_path_2[i][0]), int(drone_path_2[i][1]), yaw))

    print("drone path len: ", len(drone_path))
    print("drone_path inside:\n")
    print(drone_path)
    #input("pause ...\n\n")
    return drone_path