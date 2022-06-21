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
# 07/05/2020: added fun_dist_3d
def fun_dist_3d(loc1, loc2, loc1_z, loc2_z):
    tmp = np.sqrt(np.power((loc1[0]-loc2[0]),2) + np.power((loc1[1]-loc2[1]),2) + np.power((loc1_z-loc2_z),2))
    return tmp.item()

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

# list_turn_pts now is 3d
def getDronePath_3d(drone_init_loc, list_turn_pts=None):

    drone_path_2=[]
    #print("list turn pts in DBD getDpath",list_turn_pts)
    #input()
    for item in list_turn_pts:
        drone_path_2.append((int(item[0]), int(item[1]), 0, int(item[2]), 0))
        # put height and pitch at the end, the others' order unchanged
    #print("drone_path2 is ",drone_path_2)

    drone_path=[]

    drone_path.append((int(drone_path_2[0][0]), int(drone_path_2[0][1]), 0, int(drone_path_2[0][3]), 0))

    for i in range(1, len(drone_path_2)):

        if drone_path_2[i][3] == drone_path_2[i-1][3]:
            pitch = 0 # not the 'real' pitch, drones don't head down
            # to indicate whether go up, down or stay
        elif drone_path_2[i][3] > drone_path_2[i-1][3]:
            pitch = 90 # degree
        elif drone_path_2[i][3] < drone_path_2[i-1][3]:
            pitch = -90 # degree
        else:
            pitch = 0

        yaw = cal_yaw(drone_path_2[i], drone_path_2[i - 1])

        if int(drone_path_2[i][0]) == int(drone_path_2[i - 1][0]) \
                and int(drone_path_2[i][1]) == int(drone_path_2[i - 1][1])\
                and int(drone_path_2[i][3]) == int(drone_path_2[i - 1][3]):
            print("drone_path_2[i/i-1][3] are: ", drone_path_2[i][3], drone_path_2[i-1][3])
            continue
        else:
            #print("x in path ", i, " is: ", int(drone_path_2[i][0]))
            #print("y in path ", i, " is: ", int(drone_path_2[i][1]))
            #print("h in path ",i," is: ", int(drone_path_2[i][3]))
            #input()
            drone_path.append( (int(drone_path_2[i][0]), int(drone_path_2[i][1]), yaw, int(drone_path_2[i][3]), pitch) )
    #print("final drone path is: ", drone_path)
    #input("pause ...\n\n")
    return drone_path

def cal_yaw(drone_path_2_curr, drone_path_2_prev):
    if drone_path_2_curr[0] > drone_path_2_prev[0] and int(drone_path_2_curr[1]) == int(drone_path_2_prev[1]):
        yaw = 0
    elif drone_path_2_curr[0] < drone_path_2_prev[0] and int(drone_path_2_curr[1]) == int(drone_path_2_prev[1]):
        yaw = 180  # degree
    elif drone_path_2_curr[1] > drone_path_2_prev[1] and int(drone_path_2_curr[0]) == int(drone_path_2_prev[0]):
        yaw = 90  # degree
    elif drone_path_2_curr[1] < drone_path_2_prev[1] and int(drone_path_2_curr[0]) == int(drone_path_2_prev[0]):
        yaw = -90  # degree
    elif drone_path_2_curr[1] == drone_path_2_prev[1] and int(drone_path_2_curr[0]) == int(drone_path_2_prev[0]):
        yaw = 0
    else:
        yaw_arc = math.atan((int(drone_path_2_curr[1]) - int(drone_path_2_prev[1])) / (
                int(drone_path_2_curr[0]) - int(drone_path_2_prev[0])))
        yaw = yaw_arc * 180 / math.pi
        if drone_path_2_curr[0] > drone_path_2_prev[0]and int(drone_path_2_curr[1]) > int(drone_path_2_prev[1]):
            yaw = yaw
        elif drone_path_2_curr[0] < drone_path_2_prev[0] and int(drone_path_2_curr[1]) > int(drone_path_2_prev[1]):
            yaw = 180 + yaw  # degree
        elif drone_path_2_curr[0] < drone_path_2_prev[0] and int(drone_path_2_curr[1]) < int(drone_path_2_prev[1]):
            yaw = yaw - 180  # degree
        elif drone_path_2_curr[0] > drone_path_2_prev[0] and int(drone_path_2_curr[1]) < int(drone_path_2_prev[1]):
            yaw = yaw  # degree
        else:
            yaw = 0
    return yaw