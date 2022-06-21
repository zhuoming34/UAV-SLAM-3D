"""
Author: Honggang Zhang @ UMass Boston, honggang.zhang@umb.edu.
to simulate drones in beacon dection tasks.
based on a FastSLAM 1.0 example by Atsushi Sakai (@Atsushi_twi)
and based on an exisitng EKF filer code from an online book [cite here]
"""

"""
07/23/20:
new 3d motion model
"""

from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import ExtendedKalmanFilter, linalg
from filterpy.kalman import unscented_transform, MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import predict as kf_predict
from filterpy.kalman import update as kf_update
from filterpy.kalman import KalmanFilter as KF

from scipy.stats import entropy

import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.stats as stats
import copy

import DBD_3d_0723 as DBD
import combine_pkls_0723 as CP
# import graph_test_2 as GraphData
import pickle

import sys
from os.path import exists
from os import mkdir

# import networkx as nx
# import matplotlib.pyplot as plt
# from networkx.algorithms import approximation as approx
# import random
# from scipy.optimize import least_squares

# main
# ini stage use this:
last_stage_end = np.matrix([[], [], [], [], []])

##1st 2nd, 3rd stage...use this:
# with open('drone_act.pk', 'rb') as f:
#    last_drone = pickle.load(f)
# last_stage_end = copy.deepcopy(last_drone)

# print("last_stage_end1", last_stage_end)

# Question: why is there no process noise of landmark's estimate in FastSLAM method?
# Note that this process noise is called Q, used in the predicting step of EKF.

# ----- 03/2020, this Q is not used in my code ------
# Q = np.diag([3.0, mathmotion_model(.radians(10.0)])**2
Q = np.diag([0.10, math.radians(0.1)]) ** 2
# Q: FastSlam's algorithm's maintained covariance,
# It is the R in the thesis, which is used in updating Z_{n,t} in Eqn (3.28),
# which is called R_t in Eqn (3.28).
# In KF book, this is also called R, e.g., ukf.R, measurement covariance.

# R = np.diag([1.0, math.radians(20.0)])**2
# R_motion_control = np.diag([1.0, math.radians(20.0)])**2 * 0.1
# R_motion_control = np.diag([1.0, math.radians(20.0)])**2 * 0.5

GLOBAL_ENTROPY_LIST = []  # 04/2020
GLOBAL_VAR_LIST = []  # 04/2020

GLOBAL_LIST = []
GLOBAL_LIST_UD_DRONE = []
GLOBAL_LIST_UD_PARTICLE = []

# normal_noise_motion_control = np.array([0.02, 0.017])
# these two values are the one std of the translation speed and angle speed.
# normal_noise_motion_control = np.array([0.01, 0.017])/Zmal_noise_motion_control = np.array([0.05, 0.005])
# 0.017 radian is roughly 1 degree, as 1 degree = 3.14/180 radian.

# R: FastSlam algorithm's added noise to control input (which is the actual, noised input).
# This is different from the R in the thesis, nor the R in KF book.

# ----- 03/2020, this Qsim is not used in my code ------
# Qsim = np.diag([0.3, math.radians(2.0)])**2
# Qsim: Parameter for simulation purpose
# Qsim: noise added to the distance and angle difference between a landmark and true location of robot.

# ----- 03/2020, this Rsim is not used in my code ------
# Rsim = np.diag([0.5, math.radians(10.0)])**2
# Rsim: noise added to the acuation control input.
# Rsim: Parameter for simulation purpose

# ----- 04/2020, this is not used in my code ----
# OFFSET_YAWRATE_NOISE = 0.01

# -----------------------------
# DT = 0.1  # time tick [s]

'''
DT = 1  # time tick [s]
#SPEED = 0.5  # m/s
#MOVE_STEP = 50 # cm
#SPEED = 0.1
MOVE_STEP = 50 # cm. used in move_command()
#MOVE_STEP = 10 # cm. used in move_command()  # commented out on April 19, 2020. This was used in generating figures in the paper in March 2020.
SPEED = MOVE_STEP /100.0 / DT
#SPEED = 0.01
#MOVE_STEP = 1 # cm. used in move_command()

#----- 03/202, CAUTION: this global variable will be updated in main() function when running multiple experiments.
#     check different motion noise levels, used for prediction step of SLAM in predicting a particle's pos.
#normal_noise_motion_control = np.array([0.0, 0.0])
normal_noise_motion_control = np.array([1.0 * SPEED, 0.00001*3.14/180])
#normal_noise_motion_control = np.array([0.05, 0.1*3.14/180])
#normal_noise_motion_control = np.array([0.1, 0.1*3.14/180])
#normal_noise_motion_control = np.array([0.3, 0.1*3.14/180])

SCALE_MOTION_STD_DRONE = 1  # this adds more noise to real motion.

R_TARGET =  0.0001     # variance of target beacons measurement data in their EKFs' calculations.

#----- 03/202, CAUTION: this global variable will be updated in main() function when running multiple experiments.
RSSI_STD_TARGET = 2  # this adds noise to real RSSI data

R_ANCHOR = 0.0001

#----- 03/202, CAUTION: this global variable will be updated in main() function when running multiple experiments.
FLAG_LADDER  = 0

BEACON_COV_GUESS = np.diag([ (1 *  MOVE_STEP/100.0) ** 2, (1 *  MOVE_STEP/100.0) ** 2])
BEACON_COV_GUESS_ANCHOR = 0.001 * BEACON_COV_GUESS
#BEACON_COV_GUESS = np.diag([ 2** 2, 2 ** 2])

RND_SEED_LIST = []

show_animation = True

DEBUG = 1

#RSSI_N=2.5
#RSSI_N=2
#RSSI_A0=-45
RSSI_N=1.68
RSSI_A0=-47.29
# Should be the same as those in DroneBeaconData.py code.

'''

# ----- 03/202, CAUTION: these global variables will be updated in main() function when running multiple experiments.
GLOBAL_LIST_D_RSSI = []
GLOBAL_LIST_DIFF_D = []

# N_PARTICLE = 500  # number of particle
# N_PARTICLE = 3  # number of particle
# N_PARTICLE = 5  # number of particle
N_PARTICLE = 100  # number of particle
# N_PARTICLE = 2  # number of particle
# N_PARTICLE = 10
# -----------------------------

# SIM_TIME = 200.0  # simulation time [s]
# SIM_TIME = 10000.0  # simulation time [s]
# MAX_RANGE = 20.0  # maximum observation range
# M_DIST_TH = 2.0  # Threshold of Mahalanobis distance for data association.
# STATE_SIZE = 3  # State size [x,y,yaw]
# LM_SIZE = 2  # size of each LM [x,y]
STATE_SIZE = 3  # State size [x,y,yaw]
STATE_SIZE_3d = 5  # [x,y,yaw,h,pitch]
LM_SIZE = 3  # size of each LM [x,y,h]

LAYER = 0  # will be updated in main()
LAYER_STEP = 50  # m

PKL_HTRUE = ""  # 3d replay test

GL_LIST_LAST_B_EST = [[]]
FLAG_EST_INIT = 0
FLAG_FASTSLAM2 = 1
FLAG_KNOWN_DATA = 0

NTH = N_PARTICLE / 1.5  # Number of particle for re-sampling

LARGE_NEGATIVE = -999999.0  # used as placeholder values of landmarks.

# KF_FLAG = 0 # UKF
KF_FLAG = 1  # EKF


# a simple 1d Kalman filter
def sim_kf(x, P, u, Q, z, R):
    # x, P: state and variance of the system
    # u, Q: movement due to the process, and noise in the process
    # z, R: measurement and measurement variance
    x, P = kf_predict(x=x, P=P, u=u, Q=Q)
    x, P = kf_update(x=x, P=P, z=z, R=R)
    return x, P


# height measurement
def height(h, std):
    return h + (np.random.randn() * std)


# used in UKF
def f_radar(x, dt):
    """ state transition function for a constant velocity
    aircraft with state vector [x, velocity, altitude]'"""
    F = np.array([[1, 0], [0, 1]], dtype=float)  # 2x2 matrix
    return np.dot(F, x)


def h_radar_drone_pos(x, my_drone_pos):
    """ Measurement model. Assume that we know the location of the drone,
    which receives signal from the beacon to be located."""
    dx = x[0] - my_drone_pos[0]
    dy = x[1] - my_drone_pos[1]
    slant_range = math.sqrt(dx ** 2 + dy ** 2)
    if slant_range <= 0.002:
        slant_range = 0.002
    # result = RSSI_A0 - 10 * RSSI_N * np.log10(slant_range)
    result = slant_range
    return [result]


def HJacobian_at_drone_pos(x, my_drone_pos):
    """ compute Jacobian of H matrix at x """
    dx = x[0] - my_drone_pos[0]
    dy = x[1] - my_drone_pos[1]
    # denom = math.sqrt(dx**2 + dy**2)
    denom = math.sqrt(dx ** 2 + dy ** 2)
    if denom <= 0.00001:
        denom = 0.00001
    # print("=========in HJacobian dx, dy, denom", dx, dy, denom)
    return np.array([[dx / denom, dy / denom]])


# 3d EKF
def h_radar_drone_pos_3d(x, my_drone_pos):
    """ Measurement model. Assume that we know the location of the drone,
    which receives signal from the beacon to be located."""
    dx = x[0] - my_drone_pos[0]
    dy = x[1] - my_drone_pos[1]
    dz = x[2] - my_drone_pos[2]
    slant_range = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    if slant_range <= 0.002:
        slant_range = 0.002
    # result = RSSI_A0 - 10 * RSSI_N * np.log10(slant_range)
    result = slant_range
    return [result]


def HJacobian_at_drone_pos_3d(x, my_drone_pos):
    """ compute Jacobian of H matrix at x """
    # x: state vector of landmark position [theat_x, theta_y, theta_h]
    # my_drone_pos[0] & [1] are [x,y] from particle filter
    # my_drone_pos[2] is height [h] from KF
    dx = x[0] - my_drone_pos[0]
    dy = x[1] - my_drone_pos[1]
    dz = x[2] - my_drone_pos[2]
    # denom = math.sqrt(dx**2 + dy**2)
    denom = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    if denom <= 0.00001:
        denom = 0.00001
    # print("=========in HJacobian dx, dy, denom", dx, dy, denom)
    return np.array([[dx / denom, dy / denom, dz / denom]])


# -----------------------------------------------------------

class Drone:

    # 07/02/2020: added list_target/anchor_beacons_h(s), current_layer, max_corrd_h, all_RFID_array_h
    # list_target/anchor_beacons(s) contain [x,y] only as before
    # all_RFID_array contains only [x,y ]
    def __init__(self, id, drone_init_loc, all_RFID_array, all_RFID_array_h, R_target, R_anchor, \
                 list_target_beacon_IDs, list_anchor_beacon_IDs, \
                 list_target_beacons_xyh, list_anchor_beacons_xyh, \
                 list_target_beacons, list_anchor_beacons, \
                 list_target_beacons_h, list_anchor_beacons_h, current_layer, \
                 lmid_low, lmid_high, flag_ladder, scale_rssi_noise, scale_rssi_noise_anchors, \
                 beacon_loc_guess_array,
                 boundary_range_initial_lm_guess,
                 max_coord_x, max_coord_y, max_coord_h, active_anchor_dist_range,
                 rssi_std_step, rssi_std_est_target, rssi_std_est_anchor,
                 # -------------------------------
                 flag_restart, temp_init_loc_array, last_stage_end,  # +++
                 old_deployed_anchors_index, old_deployed_anchors_actual, old_deployed_anchors_est,
                 dict_xTrue_dict1,
                 # -------------------------------
                 normal_noise_motion_control, height_std, up_var,
                 RSSI_A0, RSSI_N, BEACON_COV_GUESS, BEACON_COV_GUESS_ANCHOR,
                 test_turn_points, DEBUG, DT, SCALE_MOTION_STD_DRONE,
                 flag_known_data,
                 pklFile_hxTrue, pklFile_hxRSSI,
                 SPEED,
                 rnd_drone_gen=np.random.RandomState(1), rnd_particle_gen=np.random.RandomState(1)):

        self.debug = DEBUG
        self.dt = DT
        self.scale_motion_std_drone = SCALE_MOTION_STD_DRONE
        self.rssi_a0 = RSSI_A0
        self.rssi_n = RSSI_N
        self.beacon_cov_guess = BEACON_COV_GUESS
        self.beacon_cov_guess_anchor = BEACON_COV_GUESS_ANCHOR

        self.sim_time = 0.0
        self.time = 0.0
        self.id = id

        # new add self-----------------------------------
        self.restart = flag_restart
        self.temp_init_loc_array = temp_init_loc_array
        self.last_stage_end = last_stage_end
        self.dict_xTrue_dict1 = dict_xTrue_dict1

        # [x, y, yaw]
        # self.drone_init_loc = np.array([0.,0.,0.,0.,0.])
        self.drone_init_loc = np.array([self.last_stage_end[0, 0], self.last_stage_end[1, 0], 0., 0., 0.])
        # print("kkk ", self.last_stage_end)
        # print("kkk ", self.last_stage_end[0])
        # print("kkkk ", self.drone_init_loc)
        self.drone_init_loc[0] = drone_init_loc[0]  # x
        self.drone_init_loc[1] = drone_init_loc[1]  # y
        self.drone_init_loc[2] = drone_init_loc[2]  # yaw
        self.drone_init_loc[3] = drone_init_loc[3]  # h
        self.drone_init_loc[4] = drone_init_loc[4]  # pitch

        self.stop_counts = 1
        self.stop_counts_max = 10  # maximum number of path segments to travel.
        self.threshhold_dist = 2  # meter.  TO BE MODIFIED.

        self.max_coord_x = max_coord_x
        self.max_coord_y = max_coord_y
        self.max_coord_h = max_coord_h

        self.active_anchor_dist_range = active_anchor_dist_range

        self.rssi_std_step = rssi_std_step  # meter
        self.rssi_std_est_target = rssi_std_est_target  # std estimated from data
        self.rssi_std_est_anchor = rssi_std_est_anchor  # std estimated from data
        self.normal_noise_motion_control = normal_noise_motion_control

        self.flag_ladder = flag_ladder
        self.scale_rssi_noise = scale_rssi_noise
        self.scale_rssi_noise_anchors = scale_rssi_noise_anchors

        self.rnd_drone_gen = rnd_drone_gen
        self.rnd_particle_gen = rnd_particle_gen

        self.rssi_actual = {}  # acutal RSSI from drone's actual position.
        self.d_rssi_actual = {}  # distance calculated based on RSSI data from drone's actual position.
        self.d_actual = {}

        self.xTrue = np.matrix(
            [[self.drone_init_loc[0]], [self.drone_init_loc[1]], [self.drone_init_loc[2] * math.pi / 180.],
             [self.drone_init_loc[3]], [self.drone_init_loc[4] * math.pi / 180.]])

        self.xActual = copy.deepcopy(self.xTrue)
        self.hxTrue = np.array([[], [], [], [], []])  # history of expected drone positions
        self.hxEst = np.array([[], [], [], [], []])  # history of estimated drone positions
        self.hxActual = np.array([[], [], [], [], []])  # history of actual drone positions

        # self.hHeightTrue = np.array([]) # history of expected drone height
        # self.hHeightEst = np.array([]) # history of estimated drone height
        # self.hHeightActual = np.array([]) # history of actual drone height

        # DON'T use np.empty, as it does not give empty array.--->>> hxTrue = np.empty((3,1), dtype=float)
        # hxEst = np.empty((3,1), dtype=float)

        self.path_index = -1  # The loop starts from a virtual location with index -1.
        self.init_loc_flag = 1

        self.est_Eula_dist_beacon_drone = 99999  # distance between estimated target beacon position and drone.
        # self.segment_dist_x = 999  # some initial big numbers as dummy case.
        # self.segment_dist_y = 999

        self.RFID = copy.deepcopy(all_RFID_array)  # make a new copy of the list and convert it to array.
        self.n_lm = self.RFID.shape[0]
        self.RFID_h = copy.deepcopy(all_RFID_array_h)

        self.boundary_range = copy.deepcopy(boundary_range_initial_lm_guess)

        self.R_target = R_target
        self.R_anchor = R_anchor
        self.list_target_beacon_IDs = list_target_beacon_IDs
        self.list_anchor_beacon_IDs = list_anchor_beacon_IDs
        self.list_target_beacons = list_target_beacons
        self.list_anchor_beacons = list_anchor_beacons

        # 07/02/2020
        self.list_target_beacons_xyh = list_target_beacons_xyh
        self.list_anchor_beacons_xyh = list_anchor_beacons_xyh
        self.list_target_beacons_h = list_target_beacons_h
        self.list_anchor_beacons_h = list_anchor_beacons_h
        self.current_layer = current_layer
        self.height_std = height_std
        self.up_var = up_var
        self.heightTrue = float(self.drone_init_loc[3] / 100)  # init height
        self.heightActual = copy.deepcopy(self.heightTrue)
        # estimated Drone height
        self.D_heightEst, self.D_heightEst_var = sim_kf(self.heightTrue, height_std ** 2, 0, up_var, self.heightActual,
                                                        height_std ** 2)
        # self.list_height_Est = [[self.D_heightEst, self.D_heightEst_var]]
        # tmp_h = current_layer
        # tmp_h_var = height_std
        # ------

        self.anchor_start_lmid = len(self.list_target_beacons)

        # assume anchor beacons always start with a beacon at position (0,0),
        # and it always comes after target beacons' indices.

        self.dict_anchor_beacons = {}  # key is location in integers, value is index in the global RFID list.

        self.target_beacon_index_list = []
        self.anchor_beacon_index_list = []
        # self.deployed_anchor_beacon_index_list = []
        self.deployed_anchor_beacon_index_list = old_deployed_anchors_index
        self.active_anchor_beacon_index_list = []  # running set, updated at each step.

        # self.RFID_deployed_anchors_actual = {}  # key: lmid, value: actual location of the anchor beacon, not the ideal location.
        self.RFID_deployed_anchors_actual = old_deployed_anchors_actual
        self.RFID_deployed_anchors_actual_h = 0  # 07/03/2020: for 3d test, place anchors on the floor
        # self.RFID_deployed_anchors_est = {}
        self.RFID_deployed_anchors_est = old_deployed_anchors_est
        self.target_beacon_reset_counter = {}

        self.hxRSSIdict = {}

        for lmid in range(0, len(self.list_target_beacons)):
            self.target_beacon_index_list.append(lmid)
            self.target_beacon_reset_counter[
                lmid] = 0  # to record the number of times a target beacon's estimate has been reset.
            self.hxRSSIdict[lmid] = np.array([])  # 04/2020

        lmid = len(self.list_target_beacons)
        for item in self.list_anchor_beacons_xyh:
            self.dict_anchor_beacons[(int(item[0]), int(item[1]), int(item[2]))] = lmid
            self.anchor_beacon_index_list.append(lmid)
            self.hxRSSIdict[lmid] = np.array([])  # 04/2020

            lmid = lmid + 1

        print("\nin Drone init(), ")
        print(self.list_anchor_beacons_xyh)  # for 2d use: self.list_anchor_beacons
        print(self.dict_anchor_beacons)  # for 2d use: self.dict_anchor_beacons
        print(self.anchor_beacon_index_list)
        print(self.deployed_anchor_beacon_index_list)

        self.beacon_loc_guess_array = np.copy(beacon_loc_guess_array)

        self.particles = [Particle(self.drone_init_loc, self.n_lm, i, self.R_target, self.R_anchor,
                                   self.list_target_beacon_IDs, self.list_anchor_beacon_IDs,
                                   self.dt, self.beacon_cov_guess,
                                   self.beacon_cov_guess_anchor) for i in range(N_PARTICLE)]

        # --------- range of landmarks to simulate [lmid_low <= and <= lmid_high-1]
        self.lmid_low = 0
        self.lmid_high = len(self.RFID)

        # Prepare to get stats of beacons.
        # robot_loc_error = []
        self.lm_est_errors = {}
        self.lm_est_x = {}
        self.lm_est_y = {}
        self.lm_est_h = {}
        self.lm_est_cov = {}
        self.lm_est_cov_diag_sum = {}
        for lmid in range(lmid_low, lmid_high):
            self.lm_est_errors[lmid] = []
            self.lm_est_x[lmid] = []
            self.lm_est_y[lmid] = []
            self.lm_est_h[lmid] = []
            self.lm_est_cov[lmid] = []
            self.lm_est_cov_diag_sum[lmid] = []

        self.speed = SPEED

        # 07/23/20
        def get_drone_path_RSSI_3d(pklFile_hxTrue, pklFile_hxRSSI):
            drone_path_3d = []  # [x,y,yaw,h,pitch]
            drone_path_RSSI_known = []
            with open(pklFile_hxTrue, 'rb') as f:  # tmp_path.pkl is saved by graph_test_2.py file.
                pkl_hxTrue = pickle.load(f)
            with open(pklFile_hxRSSI, 'rb') as f:
                pkl_hxRSSI = pickle.load(f)

            RSSI_Threshold = -65  # -65

            while True:
                origDict = {}
                for tmpidx in range(0, len(pkl_hxRSSI)):
                    # print(tmpidx, pkl_hxTrue[0, tmpidx], pkl_hxTrue[1, tmpidx], pkl_hxTrue[2, tmpidx])
                    # print(tmpidx, pkl_hxRSSI[tmpidx])
                    # if tmpidx==0 or pkl_hxRSSI[tmpidx] > -64:  # tmpidx==0 represents the initial starting point.
                    if pkl_hxRSSI[tmpidx] > RSSI_Threshold:  # tmpidx==0 represents the initial starting point.
                        origDict[tmpidx] = [(int(pkl_hxTrue[0, tmpidx] * 100), int(pkl_hxTrue[1, tmpidx] * 100),
                                             int(pkl_hxTrue[2, tmpidx] / math.pi * 180),
                                             int(pkl_hxTrue[3, tmpidx] * 100),
                                             int(pkl_hxTrue[4, tmpidx] / math.pi * 180)), round(pkl_hxRSSI[tmpidx], 3)]
                        # [ [x,y,yaw,h], RSSI ]

                origList = list(origDict.keys())
                if len(origList) > 10:
                    # input("cannot find enough good RSSIs")
                    numRepeats = int(100 / len(origList))
                    break
                else:
                    # -75
                    if RSSI_Threshold > -65:
                        RSSI_Threshold -= 3

            print("\n\nfinally, RSSI_Threshold=", RSSI_Threshold, ", len(origList)=", len(origList))
            input("pause...")

            # these two lists' indice both represents the same step numbers in the imported drone path.
            drone_path_3d.append((int(pkl_hxTrue[0, 0] * 100), int(pkl_hxTrue[1, 0] * 100),
                                  int(pkl_hxTrue[2, 0] / math.pi * 180), int(pkl_hxTrue[3, 0] * 100),
                                  int(pkl_hxTrue[4, 0] / math.pi * 180)))
            drone_path_RSSI_known.append(pkl_hxRSSI[0])

            # print("\n\n", drone_path)
            print("len(origList), numRepeats ", len(origList), numRepeats)
            for _ in range(0, numRepeats):
                np.random.seed(3)
                randomList = np.random.permutation(list(origDict.keys()))
                # print("randomList",randomList)
                for tmpidx in randomList:
                    # print("\n\n", drone_path[-1], origDict[tmpidx][0])
                    if drone_path_3d[-1][0] != origDict[tmpidx][0][0] \
                            or drone_path_3d[-1][1] != origDict[tmpidx][0][1] \
                            or drone_path_3d[-1][3] != origDict[tmpidx][0][3]:
                        drone_path_3d.append(origDict[tmpidx][0])
                        drone_path_RSSI_known.append(origDict[tmpidx][1])
                # print("drone_path ", drone_path)
                # print("NUMBER OF SAMPLES NOW: ", len(drone_path))

            if len(drone_path_3d) < 100:
                sortedList = sorted(origDict.items(), key=lambda kv: - kv[1][1])  # reversely sorted based on RSSI
                # print(sortedList)
                tmpidx = 0
                for _ in range(0, len(sortedList)):
                    # print(sortedList[tmpidx])
                    # print(sortedList[tmpidx][1][0], sortedList[tmpidx][1][1])
                    if drone_path_3d[-1][0] != sortedList[tmpidx][1][0][0] \
                            or drone_path_3d[-1][1] != sortedList[tmpidx][1][0][1] \
                            or drone_path_3d[-1][3] != sortedList[tmpidx][1][0][3]:
                        # make sure we are not repeating the same location in two consecutive moves.
                        drone_path_3d.append(sortedList[tmpidx][1][0])
                        drone_path_RSSI_known.append(sortedList[tmpidx][1][1])
                    if len(drone_path_3d) >= 100:
                        break
                    tmpidx = tmpidx + 1
            print("len of drone path 3d", len(drone_path_3d))
            print("len of RSSI known", len(drone_path_RSSI_known))
            print("smallest selected RSSI:", drone_path_RSSI_known[-1])
            # input("pause...")

            for tmpidx in range(0, len(drone_path_3d) - 1):
                if (drone_path_3d[tmpidx][0] == drone_path_3d[tmpidx + 1][0]) \
                        and (drone_path_3d[tmpidx][1] == drone_path_3d[tmpidx + 1][1]) \
                        and (drone_path_3d[tmpidx][3] == drone_path_3d[tmpidx + 1][3]):
                    print("repeating same location in consecutive moves, WRONG. ")
                    print(tmpidx, drone_path_3d[tmpidx])
                    print(tmpidx + 1, drone_path_3d[tmpidx + 1])
                    input("pause ....")
            # print(len(drone_path))
            # print(len(drone_path_RSSI_known))
            # input("pause")
            return drone_path_3d, drone_path_RSSI_known

        self.flag_known_data = flag_known_data
        self.pklFile_hxTrue = pklFile_hxTrue
        self.pklFile_hxRSSI = pklFile_hxRSSI
        # self.pklFile_hxRSSI_List = pklFile_hxRSSI_List
        # self.pklFile_hxRSSI = self.pklFile_hxRSSI_List[0]

        if self.flag_known_data == 1:

            # for tmpFile in self.pklFile_hxRSSI_List:
            #    self.drone_path, self.drone_path_RSSI_known = get_drone_path_RSSI(self.pklFile_hxTrue, tmpFile)
            self.drone_path_3d, self.drone_path_RSSI_known = get_drone_path_RSSI_3d(self.pklFile_hxTrue,
                                                                                    self.pklFile_hxRSSI)
            # self.drone_path = list(l[0:3] for l in self.drone_path_3d) # xyyaw
            # self.drone_path_h = list(l[3] for l in self.drone_path_3d) # h
            self.heightTrue = self.drone_path_3d[0][3] / 100.
            # print(len(self.drone_path_3d))
            print(self.drone_path_3d)
            print(self.heightTrue)
            input("pause")
        else:
            # if test_turn_points != None:
            self.list_turn_points = copy.deepcopy(test_turn_points)
            # -------------- pre-planned path's (x,y,yaw) coordinates and angles (in degree) ------------
            # self.drone_path = DBD.getDronePath(self.drone_init_loc, self.list_turn_points) # 2d
            # drone_init_loc is not used in DBD.getDronePath(2d/3d)
            self.drone_path_3d = DBD.getDronePath_3d(self.drone_init_loc, self.list_turn_points)  # 3d path
            print("drone_path_3d: \n", self.drone_path_3d)
            # input(" pause ")
            # now self.drone_path has [x, y, yaw, h, pitch]
        '''
        #print("\n\n drone_path")
        #print(self.drone_path)
        #input("pause ")
        '''
        # returned drone_path in cm, drone_init_loc in meters, list_turn_points in cm.

    def step_run(self, sim_time, start_time):
        self.sim_time = sim_time
        self.time = start_time

        # stop_counts, and Euler distance are used to dynamically adjust path to approach a target.
        # it has not been tested in this code yet, but it was tested in one earlier version.
        while (self.time <= self.sim_time) and (self.stop_counts <= self.stop_counts_max) \
                and self.est_Eula_dist_beacon_drone >= self.threshhold_dist:

            if self.debug == 1:
                print("\nDrone %d +++++++++++++++++++++++++++++++++++  time " % self.id, self.time, "path_index",
                      self.path_index)

            if self.init_loc_flag == 1:  # At initial location, we don't do prediction based on motion.
                # tmp_loc = (-1, -1, -1)  # current location, which is a virtual location now.
                tmp_loc = (-1, -1, -1, -1, -1)  # 07/11/20: x, y, yaw, h, pitch
                tmp_loc_next = self.drone_path_3d[0]
                self.init_loc_flag = 0
            else:
                tmp_loc = self.drone_path_3d[self.path_index]  # tmp_loc is the current expected position
                tmp_loc_next = self.drone_path_3d[self.path_index + 1]  # tmp_loc_next is the next expected position.

            if self.debug == 1:
                print("tmp_loc       ", tmp_loc, " in cm")
                print("tmp_loc_next  ", tmp_loc_next, " in cm")
                # input()
            # ----------- Calculate ud, based on expected current position and next position.
            if tmp_loc[0] == -1 or tmp_loc_next == tmp_loc:
                # If the robot does not move, i.e., at initial location, or not move (i.e., x,y,yaw all the same for both locations).
                # ud = np.matrix([0, 0]).T
                ud_3d = np.matrix([0, 0, 0, 0, 0, 0]).T
            else:
                # 07/11/20
                # flip_only = 0
                # self.speed and self.flag_known_data are not used in calc_input_new
                ud_3d, rotate_only, flip_only = calc_input_new_3d(self.time, tmp_loc_next, tmp_loc, self.speed,
                                                                  self.flag_known_data)
                # ud_3d = [v_horizon(in slant), yawrate, angle_diff, v_vertical, pitchrate, angle_diff_vertical]
                ud = ud_3d[0:3]  # [v, yawrate, angle_diff]
                # ud is a column vector np.matrix([v, yawrate]).T, yawrate is in rad/sec, NOT degree/sec.
                # No noise is added into ud in this calc_input_new function.

                if rotate_only == 1 or flip_only == 1:
                    # this should not happen if drone_path is set correctly, May 2020.
                    # in this case, ud only has angle speed, translation speed is zero.
                    if rotate_only == 1 and flip_only != 1:
                        tmp_loc = (tmp_loc[0], tmp_loc[1], tmp_loc_next[2], tmp_loc[3], tmp_loc[4])
                        print("rotate only ...")
                    elif rotate_only != 1 and flip_only == 1:
                        tmp_loc = (tmp_loc[0], tmp_loc[1], tmp_loc[2], tmp_loc[3], tmp_loc_next[4])
                        print("flip only ...")
                    else:
                        tmp_loc = (tmp_loc[0], tmp_loc[1], tmp_loc_next[2], tmp_loc[3], tmp_loc_next[4])
                        print("rotate & flip only ...")

                    # Noise added to ud in this function. need to update all particles.
                    self.particles = predict_particles_3d(self.particles, ud_3d, self.rnd_particle_gen,
                                                          self.normal_noise_motion_control, self.dt, self.D_heightEst)

                    self.xEst = calc_final_state_3d(self.particles)  # need to find estimated position of drone.

                    # Noise added to ud in this function. drone's actual position.
                    self.xActual = actual_drone_pos_3d(self.xActual, ud_3d, rotate_only, flip_only, self.rnd_drone_gen,
                                                       self.scale_motion_std_drone, self.normal_noise_motion_control,
                                                       self.dt)

                    self.heightActual = self.xActual[3, 0]
                    # after aligning angles, need to calculate translation velocity again.

                    ud_3d, rotate_only, flip_only = calc_input_new_3d(self.time, tmp_loc_next, tmp_loc, self.speed,
                                                                      self.flag_known_data)
                    # ud_3d = [v_horizon(in slant), yawrate, angle_diff, v_vertical, pitchrate, angle_diff_vertical]
                    ud = ud_3d[0:3]

                    input("pause ...")

            # ------------ Record expected position of drone at the second time point of the two times points of this step.
            # Note that we do FastSLAM based on the actual position that corresponds to the xTrue, i.e., tmp_loc_next, not tmp_loc.
            self.xTrue = np.matrix(
                [[tmp_loc_next[0] / 100.], [tmp_loc_next[1] / 100.], [tmp_loc_next[2] * math.pi / 180.],
                 [tmp_loc_next[3] / 100.], [tmp_loc_next[4] * math.pi / 180.]])
            # 20200624 add this
            self.xTrue1 = (tmp_loc_next[0] / 100., tmp_loc_next[1] / 100., tmp_loc_next[2] * math.pi / 180.,
                           tmp_loc_next[3] / 100., tmp_loc_next[4] * math.pi / 180.)

            self.heightTrue = tmp_loc_next[3] / 100.
            # print("self.xTrue\n", self.xTrue)
            # print("self.heightTrue", self.heightTrue)
            '''
            #print("ud ", ud)
            #print(ud.shape[0])
            #input("pause ...")
            '''

            # ----------------------------------------------------------------------------
            #  Find drone's actual position at the second time point of this step.
            #  This is to simulate the drone's movement to get its actual position if it moves in this step in reality.
            # ----------------------------------------------------------------------------
            if tmp_loc[0] == -1 and self.stop_counts == 1:
                self.xActual = self.xTrue
                self.heightActual = self.heightTrue

                if self.flag_ladder == 1:  # make sure the **first** anchor beacon gets deployed and active at the beginning.
                    # in future, should merge these two together,
                    # as the index_list is the keys of the RFID_deployed_anchors_actual dictionary.
                    # Here we deploy the first anchor anyway, regardless of where it actually is.
                    if self.anchor_start_lmid not in self.deployed_anchor_beacon_index_list:
                        self.deployed_anchor_beacon_index_list.append(self.anchor_start_lmid)
                    # print("deployed_anchor_beacon_index_list", deployed_anchor_beacon_index_list)

                    # deployed anchor's actual pos.
                    self.RFID_deployed_anchors_actual[self.anchor_start_lmid] = (
                    self.xActual[0, 0], self.xActual[1, 0], 0)

                    # deployed anchor's estimated pos. At the beginning it is the first anchor's actual pos at the origin.
                    self.RFID_deployed_anchors_est[self.anchor_start_lmid] = (self.xActual[0, 0], self.xActual[1, 0], 0)

                    self.active_anchor_beacon_index_list.append(self.anchor_start_lmid)

            else:
                # xActual = [x,y,yaw,h,pitch]
                self.xActual = actual_drone_pos_3d(self.xActual, ud_3d, rotate_only, flip_only, self.rnd_drone_gen,
                                                   self.scale_motion_std_drone, self.normal_noise_motion_control,
                                                   self.dt)  # drone's actual
                self.heightActual = self.xActual[3, 0]
                # print("xActual\n ", self.xActual)
                # print("heightActual ", self.heightActual, "\n")
            # ----------------------------------------------------------------------------------------------------------
            #   calcuate actual RSSI and actual distance to each beacon while this drone is at the second time point
            #   out of the two time points during this step.
            #   for target beacons, we use their true locations, as they are to be estimated.
            #   for deployed anchor beacons, we use their actual locations when deployed, NOT their ideal expected locations.
            # ---------------------------------------------------------------------------------------------------------
            tmp_x = self.xActual[0, 0]
            tmp_y = self.xActual[1, 0]
            tmp_h = self.heightActual
            # print("tmpx, tmpy, tmph", tmp_x, tmp_y, tmp_h)
            # input("tmpxyh")

            self.rssi_actual.clear()  # acutal RSSI from drone's actual position. This is unknown to Drone.
            self.d_rssi_actual.clear()  # distance calculated based on RSSI data from drone's actual position.
            self.d_actual.clear()

            for lmid in range(self.lmid_low, self.lmid_high):
                if lmid in self.target_beacon_index_list:
                    self.rssi_actual[lmid] = 0
                    if self.flag_known_data == 1:
                        self.rssi_actual[lmid] = self.drone_path_RSSI_known[
                            self.path_index + 1]  # RIGHT NOW, ONLY target beacon 0!!!
                    else:
                        # print("RFID[limd]", self.RFID[lmid])
                        # print("RFID_h[lmid]", self.RFID_h[lmid])
                        # input("RFID")

                        # self.rssi_actual[lmid] = self.rssi_actual[lmid] + \
                        #                          DBD.generate_RSSI_3d(self.RFID[lmid], self.RFID_h[lmid],
                        #                                               [tmp_x, tmp_y], tmp_h,
                        #                                               self.scale_rssi_noise,
                        #                                               self.rssi_std_step, self.rssi_std_est_target)
                        self.rssi_actual[lmid] = self.rssi_actual[lmid] + get_rssis(self.xTrue1)

                        # self.RFID[lmid] specifies the true location of target beacon lmid.
                    self.d_rssi_actual[lmid] = math.pow(10,
                                                        (self.rssi_a0 - (self.rssi_actual[lmid])) / (10 * self.rssi_n))
                    self.d_actual[lmid] = DBD.fun_dist_3d([tmp_x, tmp_y], self.RFID[lmid], tmp_h,
                                                          self.RFID_h[lmid])  # for debug purpose.

                GLOBAL_LIST_D_RSSI.append(self.d_rssi_actual[0])
                GLOBAL_LIST_DIFF_D.append(self.d_rssi_actual[0] - self.d_actual[0])

                if (self.flag_ladder == 1) and (lmid in self.deployed_anchor_beacon_index_list):
                    # Since this is for anchor beacon, we need to use the actual deployed location, not the
                    # the planned location of the anchor beacon (i.e., drone's poistion) to generate RSSI.measurements
                    # self.RFID_deployed_anchors_actual[lmid] give the actual location of a deployed anchor beacon.
                    # self.rssi_actual[lmid] = DBD.generate_RSSI_3d(self.RFID_deployed_anchors_actual[lmid],
                    #                                               self.RFID_deployed_anchors_actual_h,
                    #                                               [tmp_x, tmp_y], tmp_h,
                    #                                               self.scale_rssi_noise_anchors,
                    #                                               self.rssi_std_step, self.rssi_std_est_anchor)
                    self.rssi_actual[lmid] = get_rssis(self.xTrue1)

                    self.d_rssi_actual[lmid] = math.pow(10,
                                                        (self.rssi_a0 - (self.rssi_actual[lmid])) / (10 * self.rssi_n))
                    self.d_actual[lmid] = DBD.fun_dist_3d([tmp_x, tmp_y], self.RFID_deployed_anchors_actual[lmid],
                                                          self.heightActual, self.RFID_deployed_anchors_actual_h)
                    # print("\ndeployed anchor beacon = ", lmid, "\td_actual = %.3f" % self.d_actual[lmid])
                    # print("deployed anchor beacon = ", lmid, "\td_rssi_actual = %.3f" % self.d_rssi_actual[lmid])
                    # print("lmid ", self.rssi_actual[lmid])

            # 20200624 added this
            # store dict1
            for lmid in range(self.lmid_low, self.lmid_high):
                if self.xTrue1 not in self.dict_xTrue_dict1.keys():
                    self.dict_xTrue_dict1[self.xTrue1] = {}
                    self.dict_xTrue_dict1[self.xTrue1][lmid] = []
                else:
                    if lmid not in self.dict_xTrue_dict1[self.xTrue1]:
                        self.dict_xTrue_dict1[self.xTrue1][lmid] = []
                if lmid in self.target_beacon_index_list or lmid in self.deployed_anchor_beacon_index_list:
                    if self.rssi_actual[lmid] > -640000:  # -64.09: # within 10m 20200703 changed this to store all data
                        self.dict_xTrue_dict1[self.xTrue1][lmid].append(self.rssi_actual[lmid])

            z = np.matrix(np.zeros((0, 3)))
            for lmid in range(self.lmid_low, self.lmid_high):
                if (not (lmid in self.target_beacon_index_list)) and (
                not (lmid in self.deployed_anchor_beacon_index_list)):
                    zi = np.matrix(
                        [-1, -1, lmid])  # use placeholders, as we always need a complete z matrix for simulation.
                else:
                    zi = np.matrix([self.d_rssi_actual[lmid], 0, lmid])  # measured distance based on RSSI for lmid
                z = np.vstack((z, zi))

            # print("\n z", z.shape)
            # print(z)

            # ----------------------------------------------------------------------------
            #        Do FastSLAM
            # ----------------------------------------------------------------------------
            # if tmp_loc[0] > -1: # if the first time point of this time step is not a virtual location.
            if tmp_loc[0] >= -1:  # edited in March 2020.
                # update KF
                self.D_heightEst, self.D_heightEst_var = sim_kf(self.D_heightEst, self.D_heightEst_var,
                                                                0, self.up_var,
                                                                self.heightActual, self.height_std ** 2)

                self.particles = predict_particles_3d(self.particles, ud_3d, self.rnd_particle_gen,
                                                      self.normal_noise_motion_control, self.dt, self.D_heightEst)

                # self.boundary_range = np.matrix([[0, 0], [0, 0]]) # added in April 2020
                self.boundary_range = np.matrix([[0, 0], [0, 0], [0, 0]])  # 07/07/20
                # self.boundary_range = np.matrix([[self.max_coord_x / 2, self.max_coord_x /2 ],
                #                                 [self.max_coord_y /2, self.max_coord_y / 2]]) # added in April 2020
                # self.boundary_range = np.matrix([[0.0, self.max_coord_x], [0.0, self.max_coord_y]]) # commented out in April 2020.
                # bourndary_range specifies the area to spread initial estimated Beacon position.
                # This bourndary_range is only used for setting a target beacon's initial position (in the EKF to estimate it)
                #                   when it is first discovered, i.e., RSSI of it is received first time.
                # A anchor beacon's initial poistion (in the EFK to estimate it) is either at (0,0) when it is the first anchor,
                #                                       or the drone's itself estimated position when deploying the anchor.

                # 07/07/20: added self.heightActual
                self.particles = update_with_observation(self.particles, z, \
                                                         self.beacon_loc_guess_array, self.boundary_range, \
                                                         self.target_beacon_index_list, \
                                                         self.active_anchor_beacon_index_list, \
                                                         self.deployed_anchor_beacon_index_list, \
                                                         self.RFID_deployed_anchors_est,
                                                         self.flag_ladder,
                                                         # ---------------
                                                         self.restart, self.temp_init_loc_array,
                                                         # ---------------
                                                         (self.xActual[0, 0], self.xActual[1, 0], self.heightActual),
                                                         self.anchor_start_lmid,
                                                         self.beacon_cov_guess, self.beacon_cov_guess_anchor,
                                                         self.D_heightEst)

                # we use xActual to set a newly deployed anchor beacon's location. Note that this location is not known
                # to the drone, because the drone only knows the estimated location, but we need the actual location
                # of anchor beacons in order to calculate RSSI for simulation purpose.

                self.particles = resampling(self.particles)
            # ----------------------------------------------------------------------------

            if tmp_loc[0] == -1 and self.stop_counts == 1:
                # we use stop_counts =1, because we set xEst to xTrue only when at the very beginning if there are multiple stages.
                self.xEst = self.xTrue
                self.D_heightEst = self.heightTrue  # drone est height
                self.D_heightEst_var = height_std ** 2
                # print("tmp_loc[0]=-1 is here")
            else:
                self.xEst = calc_final_state_3d(self.particles)
                # print("\nxEst:\n", self.xEst)
                # print("D_heightEst", self.D_heightEst)

            # # write drone's actual position in pk
            # if self.restart == 1:  # at 1st stage, start write drone_act into file, at the begining of 2nd stage, use it
            #     drone_act_loc = np.matrix([[self.xActual[0, 0]], [self.xActual[1, 0]], [0], [0], [0]])
            #     with open('drone_act.pk', 'wb') as f:
            #         print("drone_act stored: ", drone_act_loc)
            #         pickle.dump(drone_act_loc, f)
            # else:
            #     drone_act_loc = np.matrix([[0], [0], [0], [0], [0]])
            #     with open('drone_act.pk', 'wb') as f:
            #         pickle.dump(drone_act_loc, f)

                # print("tmp_h is %0.2f" % tmp_h)
                # self.list_height_Est[-1][1] = D_heightEst_var # update var
                # self.list_height_Est.append([self.D_heightEst, self.D_heightEst_var])

            '''
            print("\n  xTrue, %.2f, %.2f" % (self.xTrue[0,0], self.xTrue[1,0]) )
            print("xActual, %.2f, %.2f" % (self.xActual[0,0], self.xActual[1,0]) )
            print("   xEst, %.2f, %.2f" % (self.xEst[0,0], self.xEst[1,0]) )
            for lmid in range(self.lmid_low, self.lmid_high):
                #if lmid in self.target_beacon_index_list or lmid in self.deployed_anchor_beacon_index_list:
                if lmid < self.anchor_start_lmid:
                    tmpx, tmpy, tmpx_2, tmpy_2 = self.calculate_lm_est_stats(lmid)
                    print("--- Beacon ", lmid, "( %.2f, %.2f )" % (self.RFID[lmid][0], self.RFID[lmid][1]),
                          "( %.2f, %.2f )" % (tmpx, tmpy), "( %.2f, %.2f )" % (tmpx_2, tmpy_2), " %.2f" % self.rssi_actual[lmid])
                elif (self.flag_ladder == 1) and (lmid in self.deployed_anchor_beacon_index_list):
                    tmpx, tmpy, tmpx_2, tmpy_2 = self.calculate_lm_est_stats(lmid)
                    print("--- Beacon ", lmid, "( %.2f, %.2f )" % (self.RFID[lmid][0], self.RFID[lmid][1]),
                          "( %.2f, %.2f )" % (tmpx, tmpy), "( %.2f, %.2f )" % (tmpx_2, tmpy_2), " %.2f" % self.rssi_actual[lmid])
            '''

            # -----------------------------------------------------------------------------------
            # SHOULD use the actual poistion of an anchor beacon to replace its initial RFID value.
            # if the current planned or expected location is the same as an anchor beacon's position.
            # then, deploy the anchor.
            #
            if self.flag_ladder == 1:
                newly_deployed_anchor_index_list = []
                newly_lmid = -1
                for tmppos in self.dict_anchor_beacons.keys():
                    lmid = self.dict_anchor_beacons[tmppos]

                    # if drone's current expected (planned) pos is the same as an anchor's planned location.
                    if not (lmid in self.deployed_anchor_beacon_index_list) and \
                            int(self.xTrue[0] * 100) == int(tmppos[0] * 100) and int(self.xTrue[1] * 100) == int(
                        tmppos[1] * 100):
                        # need to compare them at the level of cm.

                        # print("\nlmid", lmid, self.dict_anchor_beacons[tmppos], tmppos)
                        self.deployed_anchor_beacon_index_list.append(lmid)
                        self.RFID_deployed_anchors_actual[lmid] = (self.xActual[0, 0], self.xActual[1, 0])
                        self.RFID_deployed_anchors_est[lmid] = (self.xEst[0, 0], self.xEst[1, 0])
                        newly_deployed_anchor_index_list.append(lmid)
                        newly_lmid = lmid

                        print("newly deployed anchor lmid = ", lmid)
                        print("all deployed anchors actual and est positions")
                        for tmpkey in self.RFID_deployed_anchors_actual:
                            print("\n" + str(tmpkey) + "\t( %.2f, %.2f, %.2f )"
                                  % (self.RFID_deployed_anchors_actual[tmpkey][0],
                                     self.RFID_deployed_anchors_actual[tmpkey][1],
                                     self.RFID_deployed_anchors_actual_h) + " actual")
                            print(str(tmpkey) + "\t( %.2f, %.2f, %.2f )" % (self.RFID_deployed_anchors_est[tmpkey][0],
                                                                            self.RFID_deployed_anchors_est[tmpkey][1],
                                                                            self.RFID_deployed_anchors_actual_h) + " est")
                        # This newly deployed anchor beacons won't be added by using add_new_lm()
                        # until next iteration of the main loop.

                        # WHERE IS THE CODE TO freeze EKF estimate of deployed anchors?
                        # in update_landmark_KF(particle, z, KF_flag, beacon_loc_guess_array, dict_deployed)
                if newly_lmid > 0:
                    print("\nnewly_deployed_anchor_index_list")
                    print(newly_deployed_anchor_index_list)

            ''' I commented out reset on April 2020---------------
            if self.flag_ladder == 1 and  tmp_loc[0] > -1:  # when -1, don't reset anything.
                est_pos = (self.xEst[0, 0], self.xEst[1, 0])
                dict_reset_lms_x = {}
                dict_reset_lms_P = {}
                for lmid in self.target_beacon_index_list:
                    # if the actual distance between a drone and a target becaon is less than a threshold, we reset the estimation of that target beacon.
                    if self.target_beacon_reset_counter[lmid] >= 1: # each target beacon is only reset once.
                        continue
                    tmpx, tmpy, tmpx_2, tmpy_2 = self.calculate_lm_est_stats(lmid)
                    tmpdist = DBD.fun_dist((tmpx, tmpy), est_pos)

                    # if the estimated distance to a beacon (based on estimated drone pos and estimated beacon pos),
                    # and if estimated distance based on current measured RSSI.
                    #if  tmpdist <= 3.0 or self.d_rssi_actual[lmid] <= 3.0:
                    #if self.d_rssi_actual[lmid] <= 3.0:  # drone and a target anchor's distance
                    #if tmpdist <= 3.0:
                    if tmpdist <= -1: # this means never reset. 03/2020
                        print("\ndrone's current est", est_pos, "\nactual ", self.xActual)
                        print("self.d_rssi_actual", self.d_rssi_actual, "\nself.d_actual", self.d_actual)
                        print("\ntmpx, tmpy, tmpx_2, tmpy_2, tmpdist, for lmid = ", lmid)
                        print("( %.2f, %.2f ), ( %.2f, %.2f ), %.2f " % (tmpx, tmpy, tmpx_2, tmpy_2, tmpdist))
                        print("before reset, Beacon ", lmid, "s avg est pos: %.2f, %.2f, estimated dist %.2f and d_rssi_actual %.2f"
                              % (tmpx, tmpy, tmpdist, self.d_rssi_actual[lmid]))
                        print("\nneed to reset target beacon", lmid)

                        self.target_beacon_reset_counter[lmid] = self.target_beacon_reset_counter[lmid] + 1

                        for i in range(N_PARTICLE):
                            #tmpx = np.random.uniform(est_pos[0] - min(tmpdist,self.d_rssi_actual[lmid]), \
                            #                         est_pos[0] + min(tmpdist,self.d_rssi_actual[lmid]) )
                            #tmpy = np.random.uniform(est_pos[1] - min(tmpdist,self.d_rssi_actual[lmid]), \
                            #                         est_pos[1] + min(tmpdist,self.d_rssi_actual[lmid]))
                            tmpx = est_pos[0]
                            tmpy = est_pos[1]
                            dict_reset_lms_x[lmid] = np.array([tmpx, tmpy])
                            dict_reset_lms_P[lmid] = copy.copy(BEACON_COV_GUESS)
                            # set current_anchor_pos as the drone's current pos, which
                            # essentially says we leave a anchor drone as that place.
                            # self.particles[i].reset_particle_only(est_pos)
                            self.particles[i].reset(est_pos, dict_reset_lms_x, dict_reset_lms_P)

                        tmpx, tmpy, tmpx_2, tmpy_2 = self.calculate_lm_est_stats(lmid)
                        print("after reset Beacon ", lmid ,": ( %.2f, %.2f ) " % (tmpx, tmpy),
                              "( %.2f, %.2f )" % (tmpx_2, tmpy_2))
                        input("pause, after reset")
                #if len(dict_reset_lms_x.keys()) > 0:
                #    print("\ndict_reset_lms_x, this x is NOT a coordinate")
                #    print(dict_reset_lms_x)  # this x is not x coordiante, it includes both x and y coordinates.
                #    #input("pause....")
            '''

            if self.flag_ladder == 1:
                self.active_anchor_beacon_index_list.clear()  # running set, updated at each step.
                # for debug purpose.....
                # self.active_anchor_beacon_index_list.append(self.anchor_start_lmid)

                for lmid in self.deployed_anchor_beacon_index_list:
                    # Even though we keep updating a deployed anchor's estimate in EKF,
                    # we still use its initially deployed location as its estimated location.
                    if DBD.fun_dist_3d((self.xEst[0, 0], self.xEst[1, 0]), self.RFID_deployed_anchors_est[lmid],
                                       self.D_heightEst, self.RFID_deployed_anchors_actual_h) \
                            <= self.active_anchor_dist_range:
                        self.active_anchor_beacon_index_list.append(lmid)

                if len(self.active_anchor_beacon_index_list) < 1:
                    print("active anchor list is EMPTY")
                    print("adding the first beacon anchor.")
                    self.active_anchor_beacon_index_list.append(self.anchor_start_lmid)
                    # input("pause")

            print("\ntarget beacon index list", self.target_beacon_index_list)
            print("anchor_beacon_index_list", self.anchor_beacon_index_list)
            print("deployed_anchor_beacon_index_list", self.deployed_anchor_beacon_index_list)
            print("active anchor list", self.active_anchor_beacon_index_list)

            # ----------- BEGIN get stats --------------
            print("----------get stats ---------------------------")
            for lmid in range(self.lmid_low, self.lmid_high):
                if lmid in self.RFID_deployed_anchors_est.keys():
                    tmpx = self.RFID_deployed_anchors_est[lmid][0]
                    tmpy = self.RFID_deployed_anchors_est[lmid][1]
                    tmph = self.RFID_deployed_anchors_actual_h
                    # 07/07/20: tmph = 0
                    # for now, assume always on the floor, may be changed later
                else:
                    # tmpx, tmpy, tmpx_2, tmpy_2 = self.calculate_lm_est_stats(lmid)
                    tmpx, tmpy, tmpx_2, tmpy_2, tmph, tmph_2 = self.calculate_lm_est_stats(lmid)  # 07/07/20
                # tmph = tmp_h # 07/05/2020: assume beacons are at the same height as the drone
                # tmph = self.D_heightEst
                if lmid in self.deployed_anchor_beacon_index_list:
                    tmp = math.sqrt((tmpx - self.RFID_deployed_anchors_actual[lmid][0]) ** 2 \
                                    + (tmpy - self.RFID_deployed_anchors_actual[lmid][1]) ** 2 \
                                    + (tmph - self.RFID_deployed_anchors_actual_h) ** 2)
                else:
                    tmp = math.sqrt((tmpx - self.RFID[lmid][0]) ** 2
                                    + (tmpy - self.RFID[lmid][1]) ** 2
                                    + (tmph - self.RFID_h[lmid]) ** 2)
                    # commented on April 2020, this is because target beacons' actual positions are just in RFID[].
                # tmp_lm_cov = tmp_lm_cov / N_PARTICLE
                # tmp_lm_cov_diag_sum = tmp_lm_cov[0][0] + tmp_lm_cov[1][1]
                self.lm_est_errors[lmid].append(tmp)
                self.lm_est_x[lmid].append(tmpx)
                self.lm_est_y[lmid].append(tmpy)
                # tmph = math.fabs(tmp_h - math.sqrt(math.fabs(math.pow(self.d_actual[lmid],2) - math.pow(tmp_x-tmpx,2)- math.pow(tmp_y-tmpy,2) ) ) )
                # print("tmph",tmph)
                self.lm_est_h[lmid].append(tmph)

                # self.lm_est_cov_diag_sum[lmid].append(tmp_lm_cov_diag_sum)
                ### debug
                if lmid in self.list_target_beacons:
                    print("\nBeacon estimated lmid=", lmid_rp, "(%.2f, %.2f, %.2f)" % (self.lm_est_x[lmid][-1],
                                                                                    self.lm_est_y[lmid][-1],
                                                                                    self.lm_est_h[lmid][-1]))
                # print("\nBeacon estimated lmid=", lmid)
                # for tmp_idx in range(0, len(self.lm_est_x[lmid])):
                #      print("(%.2f, %.2f)" % (self.lm_est_x[lmid][tmp_idx], self.lm_est_y[lmid][tmp_idx]))

                if self.debug == 1:
                    if lmid in self.target_beacon_index_list or lmid in self.deployed_anchor_beacon_index_list:
                        print("lmid=", lmid_rp, self.RFID[lmid], self.RFID_h[lmid],
                              " estimated (%.2f, %.2f, %.2f)" % (tmpx, tmpy, tmph))
                        print("est distance error of Beacon %d is: %.2f" % (lmid, tmp))
                        print("rssi of Beacon %d is: %.2f" % (lmid, self.rssi_actual[lmid]))

            if self.flag_ladder == 1:
                print("\n\ndeployed_anchor beacons' actual vs est locations")
                for lmid in sorted(self.RFID_deployed_anchors_actual.keys()):
                    print("(%6.2f, %6.2f) \t (%6.2f, %6.2f) \t %6.2f" % \
                          (self.RFID_deployed_anchors_actual[lmid][0], self.RFID_deployed_anchors_actual[lmid][1], \
                           self.RFID_deployed_anchors_est[lmid][0], self.RFID_deployed_anchors_est[lmid][1], \
                           DBD.fun_dist_3d(self.RFID_deployed_anchors_actual[lmid],
                                           self.RFID_deployed_anchors_est[lmid], 0, 0)))

                print("\nactive_anchor_beacon_index_list")
                print(self.active_anchor_beacon_index_list)

            # for tmpitem in self.particles:
            ### debug
            # print("\nlm ekf's R ")
            tmpitem = self.particles[0]
            # for tmpekf in tmpitem.lmEKFs:
            #     print("\n", tmpekf.R)
            #     print(tmpekf.Q)
            #     print(tmpekf.P)
            # print("Diff_actual_vs_RSSI_D, avg \t\t std")
            # print( "%.3f, %.3f" % (np.mean(GLOBAL_LIST_DIFF_D), np.std(GLOBAL_LIST_DIFF_D) ) )
            # print("-------------------------------------")
            # ----------- END get stats --------------

            # Record history.
            self.hxTrue = np.hstack((self.hxTrue, self.xTrue))
            self.hxEst = np.hstack((self.hxEst, self.xEst))
            self.hxActual = np.hstack((self.hxActual, self.xActual))

            # 07/09/20
            # self.hHeightTrue = np.hstack((self.hHeightTrue, self.heightTrue))
            # self.hHeightEst = np.hstack((self.hHeightEst, self.D_heightEst))
            # self.hHeightActual = np.hstack((self.hHeightActual, self.heightActual))

            for lmid in range(self.lmid_low, self.lmid_high):
                if lmid < self.anchor_start_lmid:
                    self.hxRSSIdict[lmid] = np.hstack((self.hxRSSIdict[lmid], [self.rssi_actual[lmid]]))
                elif (self.flag_ladder == 1) and (lmid in self.deployed_anchor_beacon_index_list):
                    self.hxRSSIdict[lmid] = np.hstack((self.hxRSSIdict[lmid], [self.rssi_actual[lmid]]))

            self.path_index = self.path_index + 1
            self.time += self.dt

            if self.path_index > (len(self.drone_path_3d) - 2):  # time to re-plan a new path.
                return True
            else:
                return False

    def calculate_lm_est_stats(self, lmid):  # calculate estimated beacon position.

        tmp_lm_x = 0
        tmp_lm_y = 0
        tmp_lm_h = 0  # 07/04/2020
        for i in range(N_PARTICLE):
            if lmid in self.deployed_anchor_beacon_index_list or lmid in self.target_beacon_index_list:
                lm_loc = [self.particles[i].lm[lmid, 0], self.particles[i].lm[lmid, 1], self.particles[i].lm[lmid, 2]]
            else:
                # lm_loc = [0, 0]  # place holder.
                # lm_loc = [LARGE_NEGATIVE, LARGE_NEGATIVE]  # place holder.
                lm_loc = [LARGE_NEGATIVE, LARGE_NEGATIVE, LARGE_NEGATIVE]  # 3d place holder.
            tmp_lm_x = tmp_lm_x + lm_loc[0]
            tmp_lm_y = tmp_lm_y + lm_loc[1]
            tmp_lm_h = tmp_lm_h + lm_loc[2]  # 07/07/20

        lm_x_1 = tmp_lm_x / N_PARTICLE
        lm_y_1 = tmp_lm_y / N_PARTICLE
        lm_h_1 = tmp_lm_h / N_PARTICLE  # 07/07/2020

        tmp_lm_x = 0
        tmp_lm_y = 0
        tmp_lm_h = 0
        for i in range(N_PARTICLE):
            if lmid in self.deployed_anchor_beacon_index_list or lmid in self.target_beacon_index_list:
                # lm_loc = [self.particles[i].lmEKFs[lmid].x[0], self.particles[i].lmEKFs[lmid].x[1]]
                lm_loc = [self.particles[i].lmEKFs[lmid].x[0], self.particles[i].lmEKFs[lmid].x[1],
                          self.particles[i].lmEKFs[lmid].x[2]]  # 3d
            else:
                # lm_loc = [0, 0]  # place holder.
                # lm_loc = [LARGE_NEGATIVE, LARGE_NEGATIVE] # place holder.
                lm_loc = [LARGE_NEGATIVE, LARGE_NEGATIVE, LARGE_NEGATIVE]  # 3d place holder
            tmp_lm_x = tmp_lm_x + lm_loc[0]
            tmp_lm_y = tmp_lm_y + lm_loc[1]
            tmp_lm_h = tmp_lm_h + lm_loc[2]  # 07/07/20

        lm_x_2 = tmp_lm_x / N_PARTICLE
        lm_y_2 = tmp_lm_y / N_PARTICLE
        lm_h_2 = tmp_lm_h / N_PARTICLE

        return lm_x_1, lm_y_1, lm_x_2, lm_y_2, lm_h_1, lm_h_2  # 07/07/20

    def error_stats_plots(self, show_plots=False):

        dict_all_beacons_errors = {}
        for lmid in range(self.lmid_low, self.lmid_high):

            if self.flag_ladder == 0 and lmid in self.anchor_beacon_index_list:
                continue

            dict_all_beacons_errors[lmid] = [self.lm_est_errors[lmid][-1]]  # -1 is the last index

            # actual positions of beacons.
            if lmid in self.target_beacon_index_list:
                dict_all_beacons_errors[lmid].append([self.RFID[lmid][0], self.RFID[lmid][1], self.RFID_h[lmid]])
                # estimated positions of beacons.
                dict_all_beacons_errors[lmid].append([self.lm_est_x[lmid][-1],
                                                      self.lm_est_y[lmid][-1], self.lm_est_h[lmid][-1]])

            if lmid in self.anchor_beacon_index_list:
                dict_all_beacons_errors[lmid].append([self.RFID_deployed_anchors_actual[lmid][0],
                                                      self.RFID_deployed_anchors_actual[lmid][1],
                                                      self.RFID_deployed_anchors_actual_h])
                dict_all_beacons_errors[lmid].append([self.RFID_deployed_anchors_est[lmid][0],
                                                      self.RFID_deployed_anchors_est[lmid][1],
                                                      self.RFID_deployed_anchors_actual_h])

            if lmid in self.target_beacon_index_list:
                tmp_title = "Target Beacon " + str(lmid) + ", pos " + str(self.RFID[lmid])
            elif lmid in self.anchor_beacon_index_list:
                tmp_title = "Anchor Beacon " + str(lmid) + ", pos " + str(self.RFID[lmid])

            if show_plots == True:
                plt.figure(tmp_title)

                plt.subplot(3, 1, 1)
                plt.plot(self.lm_est_errors[lmid], 'x-', markersize=5)
                plt.grid(True)
                # plt.ylim(0, 20)
                plt.xlabel("Time intervals")
                plt.ylabel("Distance errors")
                plt.subplot(3, 1, 2)
                plt.plot(self.lm_est_x[lmid], 'x-', markersize=5)
                plt.plot([0, len(self.list_turn_points)], [self.RFID[lmid][0], self.RFID[lmid][0]], color='r',
                         linestyle='-',
                         linewidth=1)
                plt.grid(True)
                # plt.ylim(-5, 10)
                plt.xlabel("Time intervals")
                plt.ylabel("x ")
                plt.subplot(3, 1, 3)
                plt.plot(self.lm_est_y[lmid], 'x-', markersize=5)
                plt.plot([0, len(self.list_turn_points)], [self.RFID[lmid][1], self.RFID[lmid][1]], color='r',
                         linestyle='-',
                         linewidth=1)
                plt.grid(True)
                # plt.ylim(-5, 10)
                plt.xlabel("Time intervals")
                plt.ylabel("y ")
                plt.suptitle(tmp_title)
                plt.savefig("Beacon_" + str(lmid) + '.png')

        return dict_all_beacons_errors


class Particle:

    def __init__(self, drone_init_loc, N_LM, particle_ID, R_target, R_anchor, list_target_beacon_IDs,
                 list_anchor_beacon_IDs,
                 dt, beacon_cov_guess, beacon_cov_guess_anchor):
        # 07/07/20: added drone_init_height
        self.w = 1.0 / N_PARTICLE

        self.dt = dt
        self.beacon_cov_guess = beacon_cov_guess
        self.beacon_cov_guess_anchor = beacon_cov_guess_anchor
        # self.x = drone_init_loc[0]
        # self.y = drone_init_loc[1]
        self.x = np.random.normal(loc=drone_init_loc[0], scale=0.1)
        self.y = np.random.normal(loc=drone_init_loc[1], scale=0.1)
        self.yaw = drone_init_loc[2]
        self.h = np.random.normal(loc=drone_init_loc[3], scale=0.1)
        self.pitch = drone_init_loc[4]
        # self.x = np.random.rand()*1
        # self.y = np.random.rand()*1

        # slam2.0 added this
        self.P = np.eye(3)
        # HZ. Now remove the use of lm and lmP, instead we use each EKF's x and P.
        # landmark x-y positions, each row contains x, and y.
        # lm is [Nx3], LM_SIZE=3 for 3d: x, y ,h
        self.lm = np.matrix(np.zeros((N_LM, LM_SIZE), dtype=float)) + LARGE_NEGATIVE
        # landmark position covariance
        # [N*4x3] for 3d
        self.lmP = np.matrix(np.zeros((N_LM * LM_SIZE, LM_SIZE), dtype=float))
        # This the list of all landmarks' initial values,
        # which currently are only place holders when first constructed.
        # They will be over-written when a landmark is first encountered and recorded into a particle.
        # TO BE DONE: These two should be merged with the EKF or UKF's variables in future...

        self.ID = particle_ID  # for debug purpose.

        self.lmUKFs = []  # list of UKF filters, each for each landmark.
        self.lmEKFs = []  # list of EKF filters, each for each landmark.

        # SigmaPoints and UKF are from KF book. The UKF's parameter names might be different from
        # the ones in the thesis and in ML book.
        '''
        #print("initialize particle ", self.ID)
        #print( self.x, self.y, self.yaw, "\t", self.w)
        '''
        for i in range(0, N_LM):

            points = MerweScaledSigmaPoints(n=2, alpha=.1, beta=2., kappa=0.)
            ukf = UKF(2, 1, self.dt, fx=f_radar, hx=h_radar_drone_pos, points=points)

            # ukf.x = np.array([self.lm[i, 0], self.lm[i, 1]])
            ukf.x = np.array([LARGE_NEGATIVE, LARGE_NEGATIVE])
            # HZ. Now remove the use of lm and lmP, instead we use each EKF's x and P.

            ukf.P = copy.copy(self.beacon_cov_guess)  # The to-be-located beacon's initial position covariance.
            # It is the initial lmP in FastSLAM code.
            # It is the P in KF book (i.e., covariance of the beacon's x,y coordinates).
            # It is the initial \Sigma in (3.31) of the thesis.

            # range_std = 2 # meters
            # ukf.Q = np.diag([range_std**2])
            # ukf.Q = Q_discrete_white_noise(2, dt=dt, var=4)

            # model_noise_cov = Q_discrete_white_noise(2, dt=DT, var=0.001)
            # model_noise_cov = np.eye(2)*0.0001 # 2d
            model_noise_cov = np.eye(3) * 0.0001  # 3d
            # model_noise_cov[2][2] = height_std ** 2 # necessary?
            # No. height_std is for drone's movement, not beacon

            # Q noise. This should be small, as the beacon does not move.
            ukf.Q = copy.copy(model_noise_cov)
            # This is the Q in KF book (called process noise), and it is called system noise in ML book.
            # It is different from the Q in the original FastSLAM python code.

            # ukf.R = Q[0,0]*100
            ukf.R = np.diag([0.5])
            # This is the R noise in ML book, also in FastSLAM paper.
            # But in FastSLAM Python code, this R is referred to as Q.
            # This is called Q in the original FastSLAM python code.

            self.lmUKFs.append(ukf)

            # ekf = ExtendedKalmanFilter(dim_x=2, dim_z=1) # 2d
            # EKF for landmark position dim_x=3:[x,y,h], dim_z=1:[slant]
            ekf = ExtendedKalmanFilter(dim_x=3, dim_z=1)  # 07/07/20, 3d
            # x = [x, y, h].T, z = measurement of distance (slant range)

            # ekf.x = np.array([self.lm[i,0], self.lm[i,1]])

            # ekf.x = np.array([LARGE_NEGATIVE, LARGE_NEGATIVE]) # 2d
            ekf.x = np.array([LARGE_NEGATIVE, LARGE_NEGATIVE, LARGE_NEGATIVE])  # 07/07/2020: 3d landmark state
            # HZ. Now remove the use of lm and lmP, instead we use each EKF's x and P.

            # make these values a very large negative value, as they are just placeholders.
            # the real initial values are obtained in add new landmark function.

            # state noise
            ekf.P = copy.copy(self.beacon_cov_guess)  # check beacon_cov_guess for 2d/3d, outside of this class
            # BEACON_COV_GUESS = np.diag([(1 * MOVE_STEP / 100.0) ** 2, (1 * MOVE_STEP / 100.0) ** 2,
            #                                 (1 * UP_STEP / 100.0) ** 2])  # 07/07/20: 3x3, 3d

            # print("initialize particle ", self.ID, " lmid ", i)
            # print(ekf.x.shape)
            # print(ekf.x)
            # print(ekf.P.shape)
            # print(ekf.P)

            # ekf.F = np.eye(2)   # landmark does not move.[[1 0][0 1]] # 2d
            ekf.F = np.eye(3)  # 07/07/20, 3d

            # process noise
            ekf.Q = copy.copy(model_noise_cov)  # check model_noise_cov for 2d/3d, in this class
            ## ekf.Q = Q_discrete_white_noise(2, dt=dt, var=1)
            ## range_std = 1 # meters
            ## rk.R = np.diag([range_std**2])

            # ekf.R = copy.copy(ukf.R)   # measurement noise variance or covariance
            # measurement noise
            if i in list_target_beacon_IDs:
                ekf.R = np.diag([R_target])
            if i in list_anchor_beacon_IDs:
                ekf.R = np.diag([R_anchor])
                ekf.P = copy.copy(self.beacon_cov_guess_anchor)

            self.lmEKFs.append(ekf)

    def reset_particle_only(self, drone_new_loc):
        self.x = drone_new_loc[0]
        self.y = drone_new_loc[1]
        # self.yaw = drone_new_loc[2]

        # print("reset ....")

    def reset(self, drone_new_loc, dict_reset_LMs_x, dict_reset_LMs_P):
        # self.w = 1.0 / N_PARTICLE
        self.x = drone_new_loc[0]
        self.y = drone_new_loc[1]
        # self.yaw = drone_new_loc[2]

        # print("particle new location, ", self.x, self.y)
        for lmid in dict_reset_LMs_x.keys():
            self.lmEKFs[lmid].x = np.array([dict_reset_LMs_x[lmid][0], dict_reset_LMs_x[lmid][1]])
            self.lmEKFs[lmid].P = copy.copy(dict_reset_LMs_P[lmid])

            self.lm[lmid, 0] = self.lmEKFs[lmid].x[0]
            self.lm[lmid, 1] = self.lmEKFs[lmid].x[1]
            self.lmP[2 * lmid, 0] = self.lmEKFs[lmid].P[0, 0]
            self.lmP[2 * lmid, 1] = self.lmEKFs[lmid].P[0, 1]
            self.lmP[2 * lmid + 1, 0] = self.lmEKFs[lmid].P[1, 0]
            self.lmP[2 * lmid + 1, 1] = self.lmEKFs[lmid].P[1, 1]

            print("\n\n in reset(), landmark", lmid)
            print(self.lmEKFs[lmid].x)
            print(self.lmEKFs[lmid].P)

        # input("pause, at the end of reset() ...")

    def display(self, N_LM):
        print(
            "\nParticle " + "\t" + "%.2f" % self.x + "\t" + "%.2f" % self.y + "\t" + "%.2f" % self.yaw + "\t%.2f" % self.w)
        print("Particle's landmarks ")
        print("%.2f" % self.lm)
        print(self.lmP)
        for i in range(0, N_LM):
            print("-- ukf.Q, process noise --")
            print(self.lmUKFs[i].Q)
            print("-- ukf.R, measurement noise --")
            print(self.lmUKFs[i].R)
            print("-- ukf.x --")
            print(self.lmUKFs[i].x)
            print("-- ukf.P --")
            print(self.lmUKFs[i].P)
            print("-------------\n")


'''
ud -- noised control to robot. This should be interpreted as the actual control input in reality. 
z  -- measured dist to each landmark. 
#def fast_slam1(particles, u, z): This is the original function prototype. I just re-named u to ud.
'''


def normalize_weight(particles):
    # print("\n\n particles ")
    # for p in particles:
    #    print(p.w)
    # input("pause")

    sumw = sum([p.w for p in particles])

    try:
        for i in range(N_PARTICLE):
            particles[i].w /= sumw
    except ZeroDivisionError:
        for i in range(N_PARTICLE):
            particles[i].w = 1.0 / N_PARTICLE

        return particles

    return particles


def calc_final_state(particles):
    xEst = np.zeros((STATE_SIZE, 1))

    particles = normalize_weight(particles)

    for i in range(N_PARTICLE):
        xEst[0, 0] += particles[i].w * particles[i].x
        xEst[1, 0] += particles[i].w * particles[i].y
        xEst[2, 0] += particles[i].w * particles[i].yaw

    xEst[2, 0] = pi_2_pi(xEst[2, 0])

    return xEst


def calc_final_state_3d(particles):
    xEst = np.zeros((STATE_SIZE_3d, 1))

    particles = normalize_weight(particles)

    for i in range(N_PARTICLE):
        xEst[0, 0] += particles[i].w * particles[i].x
        xEst[1, 0] += particles[i].w * particles[i].y
        xEst[2, 0] += particles[i].w * particles[i].yaw
        xEst[3, 0] += particles[i].w * particles[i].h
        # xEst[3, 0] += particles[i].h / len(particles)
        xEst[4, 0] += particles[i].w * particles[i].pitch

    xEst[2, 0] = pi_2_pi(xEst[2, 0])

    return xEst


# 07/09/20
def predict_particles_3d(particles, u, rnd_particle_gen, normal_noise_motion_control, dt, D_height_Est):
    for i in range(N_PARTICLE):
        px = np.zeros((STATE_SIZE_3d, 1))  # px is a column vector.
        # STATE_SIZE_3d = 5 : [x,y,yaw,h,pitch]
        px[0, 0] = particles[i].x
        px[1, 0] = particles[i].y
        px[2, 0] = particles[i].yaw
        px[3, 0] = particles[i].h
        px[4, 0] = particles[i].pitch

        # if u.shape[0]==3:
        # CHECK THIS FLOAT OR INTEGER stupid case in other places, where I got puzzled.
        ud = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(6, 1)

        ud[0][0] = rnd_particle_gen.normal(loc=u[0, 0], scale=normal_noise_motion_control[0])  # speed
        ud[1][0] = rnd_particle_gen.normal(loc=u[1, 0], scale=normal_noise_motion_control[1])  # yaw
        ud[2][0] = u[2, 0]  # angle diff

        ud[3][0] = rnd_particle_gen.normal(loc=u[3, 0], scale=normal_noise_motion_control[3])
        ud[4][0] = u[4, 0]
        ud[5][0] = u[5, 0]

        GLOBAL_LIST_UD_PARTICLE.append([ud[0, 0], ud[1, 0], ud[2, 0], ud[3, 0], ud[4, 0], ud[5, 0]])

        nx = motion_model_3d(px, ud, dt)  # px is 5x1 for 3d.
        # px = [x, y, yaw, h]

        particles[i].x = nx[0, 0]
        particles[i].y = nx[1, 0]
        particles[i].yaw = nx[2, 0]
        particles[i].h = D_height_Est
        particles[i].pitch = nx[4, 0]

        # particles[i].h = drone_height # 07/07/20 # is it needed?

    return particles


def actual_drone_pos_3d(current_loc_actual, u, rotate_only, flip_only, rnd_drone_gen, scale_motion_std_drone,
                        normal_noise_motion_control, dt):
    px = np.zeros((STATE_SIZE_3d, 1))
    # prev state
    # STATE_SIZE=5 for 3d: x,y,yaw,h,pitch
    # current_loc_actual = xActual
    px[0, 0] = current_loc_actual[0]
    px[1, 0] = current_loc_actual[1]
    px[2, 0] = current_loc_actual[2]
    px[3, 0] = current_loc_actual[3]
    px[4, 0] = current_loc_actual[4]
    # if u.shape[0] == 3:
    # CHECK THIS FLOAT OR INTEGER stupid case in other places, where I got puzzled.
    ud = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(6, 1)
    # ud = [v, yaw rate, angle_diff, v_h, pitch rate?, angle_diff_h]
    # u, ud are 3X1 vectors
    if rotate_only == 1:
        ud[0][0] = 0.0
    else:
        ud[0][0] = rnd_drone_gen.normal(loc=u[0, 0], scale=normal_noise_motion_control[0] * scale_motion_std_drone)

    tmp = rnd_drone_gen.normal(loc=u[1, 0], scale=normal_noise_motion_control[1])

    ud[1][0] = tmp
    ud[2][0] = u[2, 0]
    # print("u", u)
    # print("normal_noise_motion_control",normal_noise_motion_control)
    # input()
    if flip_only == 1:
        ud[3][0] = 0.0
    else:
        ud[3][0] = rnd_drone_gen.normal(loc=u[3, 0],
                                        scale=normal_noise_motion_control[3] * scale_motion_std_drone)
    ud[4][0] = u[4, 0]
    ud[5][0] = u[5, 0]

    GLOBAL_LIST.append(tmp - u[1, 0])

    GLOBAL_LIST_UD_DRONE.append([ud[0][0], ud[1][0], ud[2][0], ud[3][0], ud[4][0], ud[5][0]])
    # next state
    nx = motion_model_3d(px, ud, dt)  # px is 5x1 vector.
    # nx[3, 0] = math.fabs(nx[3, 0])
    # print("in actual_drone_pos_3d:", nx)
    # input("  ")
    return nx


def add_new_lm(particle, z, boundary_range, BEACON_COV_GUESS):
    # r = z[0, 0]  # distance
    # b = z[0, 1]  # angle
    lmid = int(z[0, 2])

    x_range = (boundary_range[0, 0], boundary_range[0, 1])
    y_range = (boundary_range[1, 0], boundary_range[1, 1])
    h_range = (boundary_range[2, 0], boundary_range[2, 1])  # 07/07/20

    # lm, lmP have 3 columns for 3d design: x, y, h
    particle.lm[lmid, 0] = np.random.uniform(x_range[0], x_range[1])
    particle.lm[lmid, 1] = np.random.uniform(y_range[0], y_range[1])
    particle.lmP[2 * lmid, 0] = BEACON_COV_GUESS[0, 0]
    particle.lmP[2 * lmid, 1] = BEACON_COV_GUESS[0, 1]
    particle.lmP[2 * lmid + 1, 0] = BEACON_COV_GUESS[1, 0]
    particle.lmP[2 * lmid + 1, 1] = BEACON_COV_GUESS[1, 1]

    # 07/07/20: add height of new landmark
    particle.lm[lmid, 2] = np.random.uniform(h_range[0], h_range[1])
    particle.lmP[2 * lmid, 2] = BEACON_COV_GUESS[0, 2]
    particle.lmP[2 * lmid + 1, 2] = BEACON_COV_GUESS[1, 2]

    # use est coordinates form prev layer
    # check if data3d/ has old files for first run
    if FLAG_KNOWN_DATA != 1:
        prev_layer = LAYER - LAYER_STEP  # default: go up
        # print(prev_layer)
        # input()
        if exists("data3d/last_Est_xyh_layer_" + str(prev_layer) + ".pkl"):
            # print("exists")
            with open("data3d/last_Est_xyh_layer_" + str(prev_layer) + ".pkl", 'rb') as f:
                last_Est_xyh = pickle.load(f)
            particle.lm[lmid, 0] = float(last_Est_xyh[lmid][0])
            particle.lm[lmid, 1] = float(last_Est_xyh[lmid][1])
            particle.lm[lmid, 2] = float(last_Est_xyh[lmid][2])

    # HZ. Now remove the use of lm and lmP, instead of using lmEKF's x and P

    # use last est as init for replay
    if FLAG_KNOWN_DATA == 1 and FLAG_EST_INIT == 1:
        list_last_beacon_est = GL_LIST_LAST_B_EST
        # list_last_beacon_est = [[37.15, 1.1, 1.45]]
        particle.lm[lmid, 0] = list_last_beacon_est[lmid][0]
        particle.lm[lmid, 1] = list_last_beacon_est[lmid][1]
        particle.lm[lmid, 2] = list_last_beacon_est[lmid][2]
    # print("particle.lm[lmid]", particle.lm[lmid])
    # We should do these individual element assignments as x of EFK has different shape than lm (matrix)
    particle.lmEKFs[lmid].x[0] = particle.lm[lmid, 0]
    particle.lmEKFs[lmid].x[1] = particle.lm[lmid, 1]
    particle.lmEKFs[lmid].x[2] = particle.lm[lmid, 2]  # 07/07/20
    particle.lmEKFs[lmid].P = copy.copy(BEACON_COV_GUESS)

    return particle


def add_new_lm_restart(particle, z, x_coord, y_coord, z_coord, BEACON_COV_GUESS):
    # r = z[0, 0]  # distance
    # b = z[0, 1]  # angle
    lmid = int(z[0, 2])

    # lm, lmP have 3 columns for 3d design: x, y, h
    particle.lm[lmid, 0] = x_coord
    particle.lm[lmid, 1] = y_coord
    particle.lmP[2 * lmid, 0] = BEACON_COV_GUESS[0, 0]
    particle.lmP[2 * lmid, 1] = BEACON_COV_GUESS[0, 1]
    particle.lmP[2 * lmid + 1, 0] = BEACON_COV_GUESS[1, 0]
    particle.lmP[2 * lmid + 1, 1] = BEACON_COV_GUESS[1, 1]

    # 07/07/20: add height of new landmark
    particle.lm[lmid, 2] = z_coord
    particle.lmP[2 * lmid, 2] = BEACON_COV_GUESS[0, 2]
    particle.lmP[2 * lmid + 1, 2] = BEACON_COV_GUESS[1, 2]
    # HZ. Now remove the use of lm and lmP, instead of using lmEKF's x and P

    # We should do these individual element assignments as x of EFK has different shape than lm (matrix)
    particle.lmEKFs[lmid].x[0] = particle.lm[lmid, 0]
    particle.lmEKFs[lmid].x[1] = particle.lm[lmid, 1]
    particle.lmEKFs[lmid].x[2] = particle.lm[lmid, 2]  # 07/07/20
    particle.lmEKFs[lmid].P = copy.copy(BEACON_COV_GUESS)

    return particle


''' 
If used in update landmark, 
    xf (2 rows, 1 column) is the estimated location (x,y) of a landmark stored in one particle. 
        estimated in time step t-1.         
    Pf (2 rows, 2 columns) is the estimated covariance of a landmark stored in one particle.
        estimated in time step t-1. 
    Return values: 
        Hf is G_\theta_n_t. Sf is Z_{n,t}
'''


def compute_jacobians(particle, xf, Pf, Q):
    dx = xf[0, 0] - particle.x
    dy = xf[1, 0] - particle.y
    dz = 0
    # self.list_height_Actual[-1][1]
    # d2 = dx**2 + dy**2
    d2 = dx ** 2 + dy ** 2 + dz ** 2
    d = math.sqrt(d2)
    # robot's pose is the one estimated in this time step t.
    # xf (coordinate x,y) is the old one, estimated in time step t-1.

    zp = np.matrix([[d, pi_2_pi(math.atan2(dy, dx) - particle.yaw)]]).T
    # column vector (estimated distance, estimated angle)

    Hv = np.matrix([[-dx / d, -dy / d, 0.0],
                    [dy / d2, -dx / d2, -1.0]])

    Hf = np.matrix([[dx / d, dy / d],
                    [-dy / d2, dx / d2]])
    # If used in update_landmark() function, this is eqn (3.33) in the fastslam thesis.
    #   Hf is G_\theta_n_t.

    Sf = Hf * Pf * Hf.T + Q
    # If used in update_landmark() function, Pf is the covariance matrix of the landmark from t-1 time slot.
    #    Sf is Z_{n,t}, eqn (3.29). but where is Pf updated? It is done in update_KF_with_cholesky()

    return zp, Hv, Hf, Sf


''' 
    particle is one specific particle, with its robot's pose s is the one predicted in time t. 
    z is the estimated (in time t) distance/angle to one specific landmark 
    z=(distance, angle, landmark_ID)
    particles[ip] = update_landmark(particles[ip], z[iz, :], Q)
'''


def update_landmark_KF(particle, z, KF_flag, beacon_loc_guess_array, active_anchor_beacon_index_list, drone_height_est):
    # 07/07/20: added drone_height in argument
    # 07/19/20: replaced drone_height by *_est

    lm_id = int(z[0, 2])  # one landmark at a time

    # UKF
    if KF_flag == 0:
        particle.lmUKFs[lm_id].predict()
        # ukf.update( item )

        particle.lmUKFs[lm_id].update(z[0, 0], hx=h_radar_drone_pos, my_drone_pos=[particle.x, particle.y])

        # ???? This code needs to be modified if UKF is ever used,
        # because we need to element-wise copying, not just these reference copying.
        particle.lm[lm_id, :] = particle.lmUKFs[lm_id].x
        particle.lmP[2 * lm_id:2 * lm_id + 2, :] = particle.lmUKFs[lm_id].P

    # EKF
    else:
        # if lm_id in dict_deployed.keys():  # don't update deployed beacons
        #    particle.lmEKFs[lm_id].x = np.array(dict_deployed[lm_id])

        if particle.lmEKFs[lm_id].x[0] <= LARGE_NEGATIVE:
            # if at the initial virtual location.
            particle.lmEKFs[lm_id].x = np.array(beacon_loc_guess_array[lm_id])

        else:
            # This update includes those deployed anchor beacons, as we need them for driving the updates of particles.
            # Still at the end of the simulation, we use their estimated and actual positions when they are first deployed
            # as their real final estimated and actual positions.

            particle.lmEKFs[lm_id].predict()

            for _ in range(0, 10):  # Temporary ????????????????????????????
                '''particle.lmEKFs[lm_id].update(z[0,0], HJacobian_at_drone_pos, h_radar_drone_pos, \
                                              args    = [particle.x, particle.y], \
                                              hx_args = [particle.x, particle.y])'''
                particle.lmEKFs[lm_id].update(z[0, 0], HJacobian_at_drone_pos_3d, h_radar_drone_pos_3d, \
                                              args=([particle.x, particle.y, drone_height_est]), \
                                              hx_args=([particle.x, particle.y, drone_height_est]))

            '''
            I added the following in EKF.py in /usr/local/lib/python3.5/dist-packages/filterpy/kalman/.               
            # HZ. 2019.
            self.H = H
            self.PHT = PHT
            '''
        # Shapes of EKF's x might be changed for some reason during computation, and
        # reshape it to (2,) does not work for some reason, so need to do element-wise re-assignment.

        c = particle.lmEKFs[lm_id].x
        # particle.lmEKFs[lm_id].x = np.array([0.0, 0.0])
        particle.lmEKFs[lm_id].x = np.array([0.0, 0.0, 0.0])

        # if c.shape[0]==2:  # if c.shape=(2,)
        if c.shape[0] == 3:
            # if c is a row vector
            particle.lmEKFs[lm_id].x = c
            # print("c", c)
            # print(c.shape)
        else:  # if c.shape=(1,2)
            # else c is a column vector
            particle.lmEKFs[lm_id].x[0] = c[0, 0]
            particle.lmEKFs[lm_id].x[1] = c[0, 1]
            particle.lmEKFs[lm_id].x[2] = c[0, 2]
            # print(c.shape)

        particle.lm[lm_id, 0] = particle.lmEKFs[lm_id].x[0]
        particle.lm[lm_id, 1] = particle.lmEKFs[lm_id].x[1]
        particle.lm[lm_id, 2] = particle.lmEKFs[lm_id].x[2]

        particle.lmP[2 * lm_id, 0] = particle.lmEKFs[lm_id].P[0, 0]
        particle.lmP[2 * lm_id, 1] = particle.lmEKFs[lm_id].P[0, 1]
        particle.lmP[2 * lm_id, 2] = particle.lmEKFs[lm_id].P[0, 2]

        particle.lmP[2 * lm_id + 1, 0] = particle.lmEKFs[lm_id].P[1, 0]
        particle.lmP[2 * lm_id + 1, 1] = particle.lmEKFs[lm_id].P[1, 1]
        particle.lmP[2 * lm_id + 1, 2] = particle.lmEKFs[lm_id].P[1, 2]

    return particle


'''
    According to the thesis, to compute weight, we need to find Z_{n,t}, which is 
    calculated based on the landmark's (x,y) estimate and covariance estimated in time t-1.
'''


# def compute_weight(particle, z, Q):
def compute_weight(particle, z):  # See Eqn (3.37) in the thesis.
    lm_id = int(z[0, 2])

    S = particle.lmEKFs[lm_id].S
    # print("S = ", S)
    if S[0, 0] < 0.00001:
        # input("???????????????????????")
        S[0, 0] = 0.00001
    # invS = particle.lmEKFs[lm_id].SI
    invS = 1. / np.linalg.inv(S)
    # dx = np.matrix([dx[0, 0]])  # only choose the distance, not angle
    dx = particle.lmEKFs[lm_id].y

    # SHOULD USE Z_{n,t}, not covariance of landmark's coordinates.
    num = math.exp(-0.5 * dx.T * invS * dx)
    # Originally dx.T is 1x2, invS is 2x2, dx is 2x1,
    # as dx is the difference between measued z and z_prior,
    # and z has two values: distance and angle difference
    # But in our code, we only look at distance, so z only has one value,
    # so dx is 1x1, invS is also 1x1.

    # den = 2.0 * math.pi * math.sqrt(np.linalg.det(S)) # original code.
    # the formula for den is not correct.

    # print("S ", S)
    # print("invS", invS)
    den = math.sqrt(2.0 * math.pi * np.linalg.det(S))
    # print("num, den = ", num, den )
    if S < 0.00001:  # to avoid division by zero. Just set w=1.0, which makes no impact on resampling.
        w = 1.0
    else:
        w = num / den  # Eqn (3.37) in the thesis.
    # print("compute_weight ", w)
    return w


'''
    5/2019, use flag_ladder to determine whether to run simulation with ladder or not. 
    If using ladder, we add target beacon, and the anchor beacons with indices <= current anchor beacon. 

    Each particle's robot's pose is the predicted or proposal value at time t, p( s_t | s_{t-1}, u_t )
    z =[the measured distance, angle at time step t, landmark's ID].
    It updates all landmarks' estimates inside each particle, and updates the weight of each particle.  
'''


# 0717 added this to compute Gs in FastSLAM2.0
def compute_Gs(particle, lm):
    """ compute Jacobian of H matrix at x, which is Gs"""
    dx = lm[0, 0] - particle.x
    dy = lm[1, 0] - particle.y
    denom = math.sqrt(dx ** 2 + dy ** 2)
    if denom <= 0.00001:
        denom = 0.00001
    Gs = np.array([[-dx / denom, -dy / denom, 0]])

    return Gs


# fastslam2.0 adds this:---------------------------------------------------
def proposal_sampling(particle, z):
    lmid = int(z[0, 2])
    # print(particle.lm[lmid, :])
    lm = np.array(particle.lm[lmid, :]).reshape(3, 1)

    # State
    x = np.array([particle.x, particle.y, particle.yaw]).reshape(3, 1)
    x1 = x

    P = particle.P
    P_inv = np.linalg.inv(P)

    Gs = compute_Gs(particle, lm)  # Gs = np.array([[-dx / denom, -dy / denom, 0.0]])
    # print("yyy Gs ", Gs)
    Qt = copy.copy(particle.lmEKFs[lmid].S)
    Qt_inv = copy.copy(particle.lmEKFs[lmid].SI)
    # 07/20/20: added the following in EKF.py in /usr/local/lib/python3.5/dist-packages/filterpy/kalman/.
    # in def update(...)
    # L321: self.SI = linalg.inv(self.S)
    dz = np.array([float(particle.lmEKFs[lmid].y)])

    # update P:
    particle.P = np.linalg.inv(Gs.T @ Qt_inv @ Gs + P_inv)  # proposal covariance

    # print("yyy x: ",x)
    # update mu/x:
    x = x @ [1.0] + particle.P @ Gs.T @ Qt_inv @ dz  # proposal mean
    # print("yyy particle.P @ Gs.T @ Qt_inv @ dz: ", particle.P @ Gs.T @ Qt_inv @ dz)
    # print("yyy x + particle.P @ Gs.T @ Qt_inv @ dz", x)

    print("particle.P", particle.P)
    print("Gs.T", Gs.T)
    print("Qt", Qt)
    print("Qt_inv", Qt_inv)
    print("dz", dz)
    print("tt x", x)
    # print("tt x @ [1]", x @ [1.0])

    x2 = x

    particle.x = x[0]
    particle.y = x[1]
    particle.yaw = x[2]

    # if not (x1[0, 0]==x2[0] and x1[1, 0]==x2[1] and x1[2, 0]==x2[2]):
    #     print("yyy drone's pos has been updated by KFSA!")
    input("pause")
    return particle


def update_with_observation(particles, z, beacon_loc_guess_array, boundary_range, \
                            target_beacon_index_list, active_anchor_beacon_index_list,
                            deployed_anchor_beacon_index_list,
                            dict_deployed_est_pos, flag_ladder,
                            # ---------------
                            restart, temp_init_loc_array,
                            # ---------------
                            current_actual_loc, anchor_start_lmid,
                            beacon_cov_guess, beacon_cov_guess_anchor, drone_height_est):
    drone_height = current_actual_loc[2]

    lm_newly_added = []

    for iz in range(len(z[:, 0])):
        lmid = int(z[iz, 2])  # landmark iz's ID
        for ip in range(N_PARTICLE):

            if flag_ladder == 1:

                if (lmid in target_beacon_index_list) and (particles[ip].lm[lmid, 0] <= LARGE_NEGATIVE) \
                        and (particles[ip].lm[lmid, 1] <= LARGE_NEGATIVE):
                    # particles[ip] = add_new_lm(particles[ip], z[iz, :], boundary_range, beacon_cov_guess)
                    # if restart == 0:
                    #     particles[ip] = add_new_lm(particles[ip], z[iz, :], boundary_range, beacon_cov_guess)
                    # elif restart == 1:
                    #     x_coord = temp_init_loc_array[lmid][0]
                    #     y_coord = temp_init_loc_array[lmid][1]
                    #     z_coord = temp_init_loc_array[lmid][2]
                    #     particles[ip] = add_new_lm_restart(particles[ip], z[iz, :], x_coord, y_coord, z_coord,
                    #                                        beacon_cov_guess)
                    x_coord = temp_init_loc_array[lmid][0]
                    y_coord = temp_init_loc_array[lmid][1]
                    z_coord = temp_init_loc_array[lmid][2]
                    particles[ip] = add_new_lm_restart(particles[ip], z[iz, :], x_coord, y_coord, z_coord,
                                                       beacon_cov_guess)
                    lm_newly_added.append(lmid)

                if (lmid in deployed_anchor_beacon_index_list) and (particles[ip].lm[lmid, 0] <= LARGE_NEGATIVE) \
                        and (particles[ip].lm[lmid, 1] <= LARGE_NEGATIVE):
                    if lmid == anchor_start_lmid:
                        # tmp_boundary_range = np.array( [(0.0, 0.0), (0.0, 0.0)] )
                        tmp_boundary_range = np.array([(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)])
                    else:
                        tmp_boundary_range = np.array([(current_actual_loc[0], current_actual_loc[0]), \
                                                       (current_actual_loc[1], current_actual_loc[1]),
                                                       (current_actual_loc[2], current_actual_loc[2])])
                        # heightActual is passed in to current_actual_loc[2]
                    particles[ip] = add_new_lm(particles[ip], z[iz, :], tmp_boundary_range, beacon_cov_guess_anchor)
                    #  beacon_cov_guess was changed to "beacon_cov_guess_anchor"in April 23, 2020. This is a small bug.

                    lm_newly_added.append(lmid)
            else:
                if (lmid in target_beacon_index_list) and particles[ip].lm[lmid, 0] <= LARGE_NEGATIVE \
                        and particles[ip].lm[lmid, 1] <= LARGE_NEGATIVE:  # new landmark

                    # particles[ip] = add_new_lm(particles[ip], z[iz, :], boundary_range, beacon_cov_guess)
                    # if restart == 0:
                    #     particles[ip] = add_new_lm(particles[ip], z[iz, :], boundary_range, beacon_cov_guess)
                    # elif restart == 1:
                    #     x_coord = temp_init_loc_array[lmid][0]
                    #     y_coord = temp_init_loc_array[lmid][1]
                    #     z_coord = temp_init_loc_array[lmid][2]
                    #     particles[ip] = add_new_lm_restart(particles[ip], z[iz, :], x_coord, y_coord, z_coord,
                    #                                        beacon_cov_guess)
                    x_coord = temp_init_loc_array[lmid][0]
                    y_coord = temp_init_loc_array[lmid][1]
                    z_coord = temp_init_loc_array[lmid][2]
                    particles[ip] = add_new_lm_restart(particles[ip], z[iz, :], x_coord, y_coord, z_coord,
                                                       beacon_cov_guess)
                    lm_newly_added.append(lmid)

    for ip in range(N_PARTICLE):
        for iz in range(len(z[:, 0])):
            tmp_z = z[iz, :]
            lmid = int(tmp_z[0, 2])

            # if lmid in lm_newly_added:  # if the lm is newly added, skip the rest of the loop.
            #    continue

            if lmid in target_beacon_index_list:
                particles[ip] = update_landmark_KF(particles[ip], z[iz, :], KF_FLAG, beacon_loc_guess_array, \
                                                   active_anchor_beacon_index_list, drone_height_est)

            if flag_ladder == 1 and (lmid in deployed_anchor_beacon_index_list):
                particles[ip] = update_landmark_KF(particles[ip], z[iz, :], KF_FLAG, beacon_loc_guess_array, \
                                                   active_anchor_beacon_index_list, drone_height_est)

            # particles[ip] = update_landmark(particles[ip], z[iz, :], Q)
            # This updates the landmark lmid's mean coordinates (x,y) and covariance, in particle ip.
            # Use predicted s_t (i.e., robot's pose at time t) and
            # measured distance and angle z_t, i.e., r and \phi (see the paper)
            # to update landmark's estimated location (x,y)
            # In the particles[ip], each landmark's estimate is still the one obtained at time t-1, but
            # it will be updated in this function call.
            # This function will NOT modify each particle's weight.

            if FLAG_FASTSLAM2 == 1:
                # fastslam2.0 adds this:---------------------------------------------------
                # update x, y, yaw and P
                particles[ip] = proposal_sampling(particles[ip], z[iz, :])
            # -------------------------------------------------------------------------

            # only use anchor beacon to update each particle's weight.
            if flag_ladder == 1:
                # if lmid in deployed_anchor_beacon_index_list or lmid in active_anchor_beacon_index_list:
                if lmid in active_anchor_beacon_index_list:
                    # print("iz=", iz, "particle=", ip, " its pos ", particles[ip].x, particles[ip].y)
                    w = compute_weight(particles[ip], z[iz, :])

                    # TEMPORARY ???????????????????????//////
                    for tmp in range(0, 100):
                        particles[ip].w *= w
                        particles = normalize_weight(particles)
                    # print("after compute weight, particles[ip].w = ", particles[ip].w)
            else:
                if lmid in target_beacon_index_list:  # use all known Landmarks' RSSI
                    w = compute_weight(particles[ip], z[iz, :])

                    # TEMPORARY ???????????????????????//////
                    for tmp in range(0, 100):
                        particles[ip].w *= w
                        particles = normalize_weight(particles)
                    # print("\ncompute weight by using TARGET beacons, lmid ", lmid, " w ", particles[ip].w)

    # ----- For some reason, the compute_weight function call modified the lmEKFs, so we need to re-align lm with lmEKF.

    for ip in range(0, N_PARTICLE):
        # print("\nParticle ", ip)
        for iz in range(len(z[:, 0])):
            tmp_z = z[iz, :]
            lmid = int(tmp_z[0, 2])

            particles[ip].lm[lmid, 0] = particles[ip].lmEKFs[lmid].x[0]
            particles[ip].lm[lmid, 1] = particles[ip].lmEKFs[lmid].x[1]
            particles[ip].lmP[2 * lmid, 0] = particles[ip].lmEKFs[lmid].P[0, 0]
            particles[ip].lmP[2 * lmid, 1] = particles[ip].lmEKFs[lmid].P[0, 1]
            particles[ip].lmP[2 * lmid + 1, 0] = particles[ip].lmEKFs[lmid].P[1, 0]
            particles[ip].lmP[2 * lmid + 1, 1] = particles[ip].lmEKFs[lmid].P[1, 1]

    return particles


def resampling(particles):
    """
    low variance re-sampling
    """
    particles = normalize_weight(particles)

    pw = []
    for i in range(N_PARTICLE):
        pw.append(particles[i].w)

    pw = np.matrix(pw)

    Neff = 1.0 / (pw * pw.T)[0, 0]  # Effective particle number
    #  print(Neff)

    if Neff < NTH:  # resampling
        wcum = np.cumsum(pw)
        base = np.cumsum(pw * 0.0 + 1 / N_PARTICLE) - 1 / N_PARTICLE
        resampleid = base + np.random.rand(base.shape[1]) / N_PARTICLE

        inds = []
        ind = 0
        for ip in range(N_PARTICLE):
            while ((ind < wcum.shape[1] - 1) and (resampleid[0, ip] > wcum[0, ind])):
                ind += 1
            inds.append(ind)

        # HZ
        # tparticles = particles[:]
        tparticles = copy.copy(particles[:])

        for i in range(len(inds)):
            particles[i].x = tparticles[inds[i]].x
            particles[i].y = tparticles[inds[i]].y
            particles[i].yaw = tparticles[inds[i]].yaw
            particles[i].h = tparticles[inds[i]].h
            particles[i].pitch = tparticles[inds[i]].pitch
            # This kind of copy also has a problem!!!!!!!!
            # particles[i].lm = tparticles[inds[i]].lm[:, :]
            # particles[i].lmP = tparticles[inds[i]].lmP[:, :]
            particles[i].lm = copy.copy(tparticles[inds[i]].lm[:, :])
            particles[i].lmP = copy.copy(tparticles[inds[i]].lmP[:, :])

            particles[i].w = 1.0 / N_PARTICLE

            # ????? HZ. Need to find out what's going on here.
            # print("\n\n in resampling(), particles[i].lm")
            # print(particles[i].lm)
            # print(particles[i].lmP)

            # print("\n")
            # for lmid in range(0, 8): # debug purpose
            #    print(particles[i].lmEKFs[lmid].x)

    # input("\n\n!!!!!!RESAMPLING pause...")
    return particles


'''
loc_new, loc_last are in cm, integer. 
'''


def calc_input_new_3d(time, loc_new, loc_last, speed, flag_known_data):
    change_x = loc_new[0] - loc_last[0]
    change_y = loc_new[1] - loc_last[1]
    change_yaw = loc_new[2] - loc_last[2]
    change_h = loc_new[3] - loc_last[3]
    change_pitch = loc_new[4] - loc_last[4]
    # 07/11/20: no pitch added in original loc

    rotate_only = 0
    flip_only = 0

    # 07/07/20, new elements for 3d
    # change_h = loc_new[3] - loc_last[3]
    # change_pitch = loc_new[4] - loc_last[4]
    # pitch_state = 0 # 0 = stay, 1 = up, -1 = down

    if time < 0.0:
        v_r = 0.0  # # speed in slant(or xy)-dir
        yawrate = 0.0
        angle_diff = 0.0  # horizontal angle diff
        v_h = 0.0  # # speed in h-dir
        pitchrate = 0.0
        angle_diff_v = 0.0  # vertical angle diff
    # elif int(change_yaw) != 0 and ( math.fabs(change_x) < 0.0001 and math.fabs(change_y) < 0.00001 ):  # only rotating.
    elif int(change_yaw) != 0 and int(change_pitch) == 0 and (
            change_x == 0 and change_y == 0 and change_h == 0):  # only rotating.
        v_r = 0
        yawrate = change_yaw * math.pi / 180.
        angle_diff = 0.0
        v_h = 0.0  # vertical v
        pitchrate = 0.0
        angle_diff_v = 0.0
        rotate_only = 1
    elif int(change_yaw) == 0 and int(change_pitch) != 0 and (
            change_x == 0 and change_y == 0 and change_h == 0):  # only flipping.
        v_r = 0
        yawrate = 0.0
        angle_diff = 0.0
        v_h = 0.0  # vertical v
        pitchrate = change_pitch * math.pi / 180.
        angle_diff_v = 0.0
        flip_only = 1
    elif int(change_yaw) != 0 and int(change_pitch) != 0 and (
            change_x == 0 and change_y == 0 and change_h == 0):  # only flipping.
        v_r = 0
        yawrate = change_yaw * math.pi / 180.
        angle_diff = 0.0
        v_h = 0.0  # vertical v
        pitchrate = change_pitch * math.pi / 180.
        angle_diff_v = 0.0
        rotate_only = 1
        flip_only = 1
    else:
        v_r = math.sqrt((change_x / 100.0) ** 2 + (change_y / 100.0) ** 2)
        yawrate = change_yaw * math.pi / 180.
        angle_diff = math.atan2(change_y, change_x)

        pitchrate = change_pitch * math.pi / 180.
        change_r = math.sqrt(change_x ** 2 + change_y ** 2)
        angle_diff_v = math.atan2(change_h, change_r)
        # v_h = v_r * math.tan(angle_diff_v)
        v_h = change_h / 100
        # this only shows the angle difference meansured between two points.
        # e.g., (0,0) and (2,1), angle difference is atan2(2-0, 1-0)
    # print("vh",v_h)
    u = np.matrix([v_r, yawrate, angle_diff, v_h, pitchrate, angle_diff_v]).T

    return u, rotate_only, flip_only


def motion_model_3d(x, u, dt):
    # x, u are 5x1 column vector, for 3d
    # x = [x, y, yaw, h, pitch]
    # u = intermediate control input, [speed, yawrate, angle diff, pitchrate, vertical angle diff]
    # u = intermediate control input, [speed, yawrate, angle diff, v_h, pitchrate, vertical angle diff]

    F = np.matrix([[1.0, 0, 0, 0, 0],
                   [0, 1.0, 0, 0, 0],
                   [0, 0, 1.0, 0, 0],
                   [0, 0, 0, 1.0, 0],
                   [0, 0, 0, 0, 1.0]])

    v_r = u[0, 0]  # speed in slant(or xy)-dir
    change_yaw = u[1, 0]  # yaw rate
    angle_diff = u[2, 0]  # horizontal angle in rad

    # angle_xy = x[2, 0]  # horizontal angle in rad


    v_h = u[3, 0]  # speed in h-dir
    change_pitch = u[4, 0]  # pitch rate
    angle_diff_v = u[5, 0]  # vertical angle difference

    tx = dt * math.cos(angle_diff)  # part of dt in x-dir
    ty = dt * math.sin(angle_diff)  # part of dt in y-dir


    tr = math.sqrt(tx ** 2 + ty ** 2)
    th = tr * math.tan(angle_diff_v)

    u_short = np.matrix([v_r, change_yaw, v_h, change_pitch]).T  # translated control input

    B = np.matrix([[tx, 0.0, 0.0, 0.0],
                   [ty, 0.0, 0.0, 0.0],
                   [0.0, dt, 0.0, 0.0],
                   [0.0, 0.0, dt, 0.0],
                   [0.0, 0.0, 0.0, dt]])
    # Bu = [dx, dy, d_yaw, dh, d_pitch].T
    # yaw = s_theta in FastSLAM paper

    # [5x1] = [5x5]*[5x1] + [5x4]*[4x1] # 07/11/20
    x = F * x + B * u_short

    # print("in motion model 3d, x is:", x)

    x[2, 0] = pi_2_pi(x[2, 0])
    x[4, 0] = pi_2_pi(x[4, 0])

    return x


def pi_2_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


# updated in 04/2020.
# This will generate a sequence of steps that move along x axis first, then move along y axis.
def move_command_3d(current_loc, x_distance, y_distance, h_distance, move_step,
                    up_step):  # all in meters, each step must be in 50cm.
    list_turn_points = []
    # print(x_distance, y_distance)
    if round(math.fabs(x_distance * 100)) == 0 and round(math.fabs(y_distance * 100)) == 0 \
            and round(math.fabs(h_distance * 100)) == 0:
        input("pause... something wrong, both x_distance and y_distance are zero in move_command()")

    if x_distance > 0:
        step = move_step  # in cm.t
        extra = 1
    else:
        step = - move_step  # in cm.
        extra = -1
    for i in range(step, x_distance * 100 + extra, step):
        # use step as initial j, in order to avoid repeating the last point in the previous loop.
        tmp_x = current_loc[0] * 100 + i
        tmp_y = current_loc[1] * 100
        tmp_h = current_loc[2] * 100
        list_turn_points.append((tmp_x, tmp_y, tmp_h))

    if y_distance > 0:
        step = move_step  # in cm.
        extra = 1
    else:
        step = - move_step  # in cm.
        extra = -1
    for j in range(step, y_distance * 100 + extra, step):
        # use step as initial j, in order to avoid repeating the last point in the previous loop.
        tmp_x = current_loc[0] * 100
        tmp_y = current_loc[1] * 100 + j
        tmp_h = current_loc[2] * 100
        list_turn_points.append((tmp_x, tmp_y, tmp_h))

    if h_distance > 0:
        step = up_step  # in cm.
        extra = 1
    else:
        step = - up_step  # in cm.
        extra = -1
    for k in range(0, h_distance * 100 + extra, step):
        tmp_x = current_loc[0] * 100
        tmp_y = current_loc[1] * 100
        tmp_h = current_loc[2] * 100 + k
        list_turn_points.append((tmp_x, tmp_y, tmp_h))

    # print("list_turn_points in move command",list_turn_points)
    return list_turn_points  # in cm.

def move_dup(one_list):
 temp_list = []
 for one in one_list:
     if one not in temp_list:
         temp_list.append(one)
 return temp_list

def get_rssis(pos):

 # with open('dict_xTrue_RSSI.pk', 'rb') as f:
 #20200610 changed this
 with open('dict_xTrue_dict1.pk', 'rb') as f:
     dict_xTrue_dict1 = pickle.load(f)

 for item in dict_xTrue_dict1.keys():
     if item[0]==pos[0] and item[1]==pos[1]:
         rssi = np.median(dict_xTrue_dict1[item][lmid_rp])

 return rssi

# tttt turn points
def get_turn_points_3d_init():
    # 0722: start_x/y should be 0, start_h should be current layer if scan layer by layer
    test_turn_points = []

    target_num = lmid_rp

    with open('dict_xTrue_dict1.pk', 'rb') as f:
        dict_xTrue_dict1 = pickle.load(f)

    # has cluster loc------------------------------------
    with open('dict_path_stage0.pk', 'rb') as f:
        dict_path = pickle.load(f)
    temp_path = dict_path[target_num]
    for i in range(0, len(temp_path)):
        # if np.median(dict_xTrue_dict1[temp_path[i]][target_num]) > rssi_threshold:  # hhh
            x = dict_path[target_num][i][0] * 100
            y = dict_path[target_num][i][1] * 100
            z = dict_path[target_num][i][2] * 100
            pos = (x, y, z)
            test_turn_points.append(pos)
            # test_turn_points=test_turn_points+[(x, y, z)]

    temp = move_dup(test_turn_points)
    np.random.seed(1)
    np.random.shuffle(temp)
    temp.insert(0, (0, 0, 0))
    test_turn_points_random = temp

    print("kkk turn points ", test_turn_points_random)

    return test_turn_points_random


def get_turn_points_3d(target_num):
    test_turn_points = []

    with open('dict_path_stage0.pk', 'rb') as f:
        dict_path_stage0 = pickle.load(f)
    #print(dict_path_stage0)
    #print(len(dict_path_stage0))
    #print(len(dict_path_stage0[0]))
    # has cluster loc------------------------------------
    with open('dict_path.pk', 'rb') as f:
        dict_path = pickle.load(f)
    #print(dict_path)
    #print(len(dict_path))
    #print(len(dict_path[0]))
    #input()

    temp_path = dict_path_stage0[target_num] + dict_path[target_num]
    #print(len(temp_path))
    #print(temp_path)
    #input()
    for i in range(0, len(temp_path)):
        #print(i)
        #print(temp_path[target_num])
        #print(temp_path[target_num][i][0],temp_path[target_num][i][1],temp_path[target_num][i][2])
        #input()
        x = int(temp_path[i][0] * 100)
        y = int(temp_path[i][1] * 100)
        z = int(temp_path[i][2] * 100)
        test_turn_points.append((x, y, z))

    temp = move_dup(test_turn_points)
    np.random.seed(1)  #s1
    np.random.shuffle(temp)
    temp.insert(0, (0, 0, 0))
    test_turn_points_random = temp

    print("kkk turn points ", test_turn_points_random)
    #input()
    return test_turn_points_random


'''
run a FastSLAM with an starting point and a trajectory, and a sequence of RSSIs. 
#def main():
'''


# 07/02/2020 modified for 3d
# added max_coord_h, current_layer, list_target/anchor_beacons_h(s), height_std, up_var,
# !!!!!!list_target_beacons & list_anchor_beacons below contain [x, y] only!!!!!!
def FastSLAM_Run(RSSI_N, RSSI_A0, RND_SEED_LIST, flag_graph_data, FLAG_LADDER, DEBUG, show_animation,
                 R_TARGET, R_ANCHOR, RSSI_STD_TARGET, RSSI_STD_ANCHOR,
                 scale_rssi_noise, scale_rssi_noise_anchors,
                 MOVE_STEP, UP_STEP, DT, SPEED, normal_noise_motion_control,
                 SCALE_MOTION_STD_DRONE, height_std, up_var,
                 max_coord_x, max_coord_y, max_coord_h, current_layer, start,
                 list_target_beacons_xyh, list_target_beacons, list_target_beacons_h,
                 list_anchor_beacons_xyh, list_anchor_beacons, list_anchor_beacons_h,
                 mid_pt_list_3d, beacon_loc_guess_array, BEACON_COV_GUESS, BEACON_COV_GUESS_ANCHOR,
                 flag_known_data, pklFile_hxTrue, pklFile_hxRSSI, outputFolder, lmid_rp):
    print(__file__ + " start!!")
    flag_ladder = FLAG_LADDER

    # rssi_std_step = 1 # meter, threshold
    rssi_std_step = 0  # meter, threshold. 03/2020. # TO BE PUT INTO THE PARAMETER LIST!!!!
    rssi_std_est_target = RSSI_STD_TARGET
    rssi_std_est_anchor = RSSI_STD_ANCHOR

    # active_anchor_dist_range = 6.0 # if a deployed beacon is within this distance away from drone, add it as active.
    active_anchor_dist_range = 100.0  # TO BE PUT INTO THE PARAMETER LIST!!!!

    rnd_seed_list = RND_SEED_LIST
#s1
    if flag_restart == 1:
        temp_init_loc_array = [(0.84, 11.88, 1.63),  # 0 ++
                               (10.74, 6.25, 2.05),  # 1 ++
                               (39.71, 0, 1.23),  # 2
                               (31.07, 13.74, 2.32),  # 3
                               (19.48, 18.95, 2.54),  # 4--
                               (5.29, 24.56, 0.29),  # 5 ++
                               (40, 24.22, 0.39),  # 6
                               (26.00, 29.10, 0.14),  # 7--
                               (1.84, 39.30, 1.81),  # 8 ++
                               (23.10, 39.16, 0.94),  # 9--
                               ]  # 3
        # with open('drone_act.pk', 'rb') as f:
        #     last_drone = pickle.load(f)
        # last_stage_end = copy.deepcopy(last_drone)
        last_stage_end = np.matrix([[0.], [0.], [0], [0.], [0]]) # offline always 000
    else:
        temp_init_loc_array_all = [(0.9, 11.2, 1.5),  # 0
                               (10.2, 7.8, 1.56),  # 1
                               (37.8, 1.4, 1.44),  # 2
                               (30.3, 11.7, 1.42),  # 3
                               (19.7, 19.6, 1.56),  # 4
                               (6.9, 22.9, 1.28),  # 5
                               (39.4, 23, 1.47),  # 6
                               (28.2, 29.6, 1.42),  # 7
                               (1.9, 38.2, 1.53),  # 8
                               (22.1, 39.1, 1.5),  # 9
                               ]  # center
        temp_init_loc_array = [temp_init_loc_array_all[lmid_rp]]
        last_stage_end = np.matrix([[0.], [0.], [0], [0.], [0]])
    print("temp_init_loc_array", temp_init_loc_array)
    print("last_stage_end", last_stage_end)

    RFID = np.array(list_target_beacons + list_anchor_beacons)  # concatenate the two lists.
    RFID_xyh = np.array(list_target_beacons_xyh + list_anchor_beacons_xyh)
    RFID_h = np.array(list_target_beacons_h + list_anchor_beacons_h)
    N_LM = RFID.shape[0]

    # --------- range of landmark indices to simulate [lmid_low <= and <= lmid_high-1]
    lmid_low = 0
    lmid_high = len(RFID)

    drone_dict = {}  # dictionary collection of all drones
    drone_end_dict = {}  # flag to show if a drone has finished its path.

    if flag_graph_data == 0:
        # test_turn_points = get_turn_points(mid_pt_list, 0, 0, MOVE_STEP)
        if flag_restart == 0:
            test_turn_points = get_turn_points_3d_init()
        else:
            start_x = start[0]
            start_y = start[1]
            test_turn_points = get_turn_points_3d(lmid_rp)
        # start_x and start_y both in meters.
        # print("mid pt list 3d is:", mid_pt_list_3d)
        # test_turn_points = get_turn_points_3d_init(mid_pt_list_3d, 0, 0, 0, MOVE_STEP, UP_STEP)  # start_x and start_y both in meters.
        # print("test_turn_points", test_turn_points)
        # input("pause")
    else:
        # tmp_graph, tmp_tree, tmp_path = GraphData.get_graph_data(20, 20, 0.2, 0.1)
        with open('tmp_path.pkl', 'rb') as f:  # tmp_path.pkl is saved by graph_test_2.py file.
            tmp_path = pickle.load(f)
        test_turn_points = []
        for item in tmp_path:
            test_turn_points.append((item[0] * 50, item[1] * 50))  # WHy times 50 here? April 2020.

    # TO be added into this function's parameter list later.
    list_drone_init_loc = [(test_turn_points[0][0], test_turn_points[0][1], 0, test_turn_points[0][2], 0)]
    # 0722: x, y, yaw, h, pitch
    num_drones = len(list_drone_init_loc)  # TO be made more flexible. 04/2020.

    # give a square area in which we uniformly at random to choose initial guess poition of a landmark.
    # boundary_initial_lm_guess = np.matrix([[0.0, float(max_coord_x)], [0.0, float(max_coord_y)]])
    # 3d boudary_init
    boundary_initial_lm_guess = np.matrix([[0.0, float(max_coord_x)],
                                           [0.0, float(max_coord_y)],
                                           [0.0, float(max_coord_h)]])
    all_beacon_errors = []
    all_drone_locs_est = []  # estiamted last pos
    all_drone_locs_exp = []  # expected last pos
    all_drone_locs_actual = []  # actual last pos
    all_drone_est_actual_errors = []  # error between estimated pos and actual pos
    all_drone_est_expect_errors = []  # error between estimated pos and expected pos.

    all_drone_height_actual = []  # actual last height
    all_drone_height_exp = []
    all_drone_height_est = []

    # ----- start simulation -----
    for tmpseed in rnd_seed_list:
        np.random.seed(tmpseed)
        start_time = 0
        sim_time = start_time + DT
        tmp_index = 0

        for drone_id in range(0, num_drones):
            print("list targets, list anchors", list_target_beacons_xyh, list_anchor_beacons_xyh)
            list_target_beacon_IDs = list(range(0, len(list_target_beacons)))
            list_anchor_beacon_IDs = list(range(len(list_target_beacons), N_LM))
            print("list_target_beacon_IDs, and list_anchor_beacon_IDs")
            print(list_target_beacon_IDs)
            print(list_anchor_beacon_IDs)
            # input("pause")

            # rnd_drone_gen: random number generator to generate drone's ud.
            # rnd_particle_gen: random number generator to generate rnd nums for SLAM.
            drone_dict[drone_id] = Drone(drone_id, list_drone_init_loc[drone_id], RFID, RFID_h, \
                                         R_TARGET, R_ANCHOR, \
                                         list_target_beacon_IDs, list_anchor_beacon_IDs, \
                                         list_target_beacons_xyh, list_anchor_beacons_xyh, \
                                         list_target_beacons, list_anchor_beacons, \
                                         list_target_beacons_h, list_anchor_beacons_h, current_layer, \
                                         lmid_low, lmid_high, flag_ladder, scale_rssi_noise, scale_rssi_noise_anchors, \
                                         beacon_loc_guess_array, boundary_initial_lm_guess, \
                                         max_coord_x, max_coord_y, max_coord_h, active_anchor_dist_range,
                                         rssi_std_step, rssi_std_est_target, rssi_std_est_anchor,
                                         # -------------------------------
                                         flag_restart, temp_init_loc_array, last_stage_end,  # +++
                                         old_deployed_anchors_index, old_deployed_anchors_actual,
                                         old_deployed_anchors_est,
                                         dict_xTrue_dict1,
                                         # -------------------------------
                                         normal_noise_motion_control, height_std, up_var,
                                         RSSI_A0, RSSI_N, BEACON_COV_GUESS, BEACON_COV_GUESS_ANCHOR,
                                         test_turn_points,
                                         DEBUG, DT, SCALE_MOTION_STD_DRONE,
                                         flag_known_data, pklFile_hxTrue, pklFile_hxRSSI,
                                         SPEED, rnd_drone_gen=np.random.RandomState(tmpseed),
                                         rnd_particle_gen=np.random.RandomState(tmpseed))
            drone_end_dict[drone_id] = False

        while True:
            for drone_id in range(0, num_drones):
                if not drone_end_dict[drone_id]:
                    drone_end_dict[drone_id] = drone_dict[drone_id].step_run(sim_time, start_time)
                    ### debug
                    # print("\n ******** after each step_run(), Drone ID %d, lm_est_errors" % drone_id)
                    # for lmid in range(lmid_low, lmid_high):
                    #     print("lmid %d, (%.2f, %.2f, %.2f), %.2f " % (lmid,
                    #                                                   drone_dict[drone_id].lm_est_x[lmid][-1],
                    #                                                   drone_dict[drone_id].lm_est_y[lmid][-1],
                    #                                                   drone_dict[drone_id].lm_est_h[lmid][-1],
                    #                                                   drone_dict[drone_id].lm_est_errors[lmid][tmp_index] ) )

            # # 20200624 added this to store dict_xTrue_RSSI into pk file
            # for drone_id in range(0, num_drones):
            #     with open('dict_xTrue_dict1.pk', 'wb') as f:
            #         pickle.dump(drone_dict[drone_id].dict_xTrue_dict1, f)
            #     f.close()

            if show_animation:
                if flag_known_data == 1:
                    plt.figure("Drone Trajectory (m) for target %d" % lmid_rp)
                else:
                    plt.figure("Drone Trajectory (m) at layer %.2f m" % current_layer)
                plt.cla()
                ex_bound_x = int(max_coord_x / 5)
                ex_bound_y = int(max_coord_y / 5)
                plt.xlim(-ex_bound_x, max_coord_x + ex_bound_x)
                plt.ylim(-ex_bound_y, max_coord_y + ex_bound_y)
                plt.grid(True)
                # plt.plot(RFID[:, 0], RFID[:, 1], "ok", markersize=8)

                # plot beacons' real poitions.
                for lmid in range(lmid_low, lmid_high):
                    if flag_ladder == 0 and lmid >= len(list_target_beacons):
                        continue
                    plt.plot(RFID[lmid][0], RFID[lmid][1], "ok", markersize=4)
                    if flag_known_data == 1:
                        plt.text(RFID[lmid][0], RFID[lmid][1], "B_%d" % (lmid_rp))
                    else:
                        plt.text(RFID[lmid][0], RFID[lmid][1], "B_%d" % (lmid))

                # plot all drones' estimated positins.
                for drone_id in range(0, num_drones):
                    for i in range(N_PARTICLE):
                        plt.plot(drone_dict[drone_id].particles[i].x, drone_dict[drone_id].particles[i].y, "c.",
                                 markersize=1)

                # re-draw history of each drone in each iteration.
                for drone_id in range(0, num_drones):
                    plt.plot(np.array(drone_dict[drone_id].hxTrue[0, :]).flatten(),
                             np.array(drone_dict[drone_id].hxTrue[1, :]).flatten(), "c^", markersize=1)
                    plt.plot(np.array(drone_dict[drone_id].hxEst[0, :]).flatten(),
                             np.array(drone_dict[drone_id].hxEst[1, :]).flatten(), "bx", markersize=1)
                    plt.plot(np.array(drone_dict[drone_id].hxEst[0, -1]).flatten(),
                             np.array(drone_dict[drone_id].hxEst[1, -1]).flatten(), "bo", markersize=6)
                    plt.plot(np.array(drone_dict[drone_id].hxActual[0, :]).flatten(),
                             np.array(drone_dict[drone_id].hxActual[1, :]).flatten(), "mo", markersize=1)
                    plt.plot(np.array(drone_dict[drone_id].hxActual[0, -1]).flatten(),
                             np.array(drone_dict[drone_id].hxActual[1, -1]).flatten(), "mo", markersize=6)

                for lmid in range(lmid_low, lmid_high):
                    if flag_ladder == 0 and lmid >= len(list_target_beacons):
                        continue
                    for drone_id in range(0, num_drones):
                        if not drone_end_dict[drone_id]:
                            tmpx = drone_dict[drone_id].lm_est_x[lmid][tmp_index]
                            tmpy = drone_dict[drone_id].lm_est_y[lmid][tmp_index]
                        else:  # if this drone has already finished, we re-draw its last position.

                            last_index = len(drone_dict[drone_id].lm_est_x[lmid]) - 1
                            tmpx = drone_dict[drone_id].lm_est_x[lmid][last_index]
                            tmpy = drone_dict[drone_id].lm_est_y[lmid][last_index]

                        for draw_tmp_idx in range(0, len(drone_dict[drone_id].lm_est_x[lmid])):
                            tmpx = drone_dict[drone_id].lm_est_x[lmid][draw_tmp_idx]
                            tmpy = drone_dict[drone_id].lm_est_y[lmid][draw_tmp_idx]
                            plt.plot(tmpx, tmpy, "go", markersize=1)
                            # plt.text(tmpx+0.1, tmpy+0.1, "%d" % draw_tmp_idx) # annotate it with step sequence number

                        tmpx = drone_dict[drone_id].lm_est_x[lmid][-1]
                        tmpy = drone_dict[drone_id].lm_est_y[lmid][-1]
                        plt.plot(tmpx, tmpy, "ro", markersize=8)

                plt.grid(True)
                plt.pause(0.001)
                # plt.show()
                plt.draw()

            tmp = 0
            for drone_id in range(0, num_drones):
                if drone_end_dict[drone_id]:
                    tmp = tmp + 1
            if tmp >= num_drones:
                break

            tmp_index = tmp_index + 1
            start_time = start_time + DT
            sim_time = start_time + DT
            # input("pause...")

        # ------- end while loop --------------------

        drone_id = 0
        dict_all_beacons_errors = drone_dict[drone_id].error_stats_plots(show_plots=False)

        all_beacon_errors.append(dict_all_beacons_errors)
        all_drone_locs_actual.append(drone_dict[drone_id].xActual)
        all_drone_locs_est.append(drone_dict[drone_id].xEst)
        all_drone_locs_exp.append(drone_dict[drone_id].xTrue)
        all_drone_est_actual_errors.append(DBD.fun_dist_3d(drone_dict[drone_id].xEst, drone_dict[drone_id].xActual,
                                                           drone_dict[drone_id].D_heightEst,
                                                           drone_dict[drone_id].heightActual))
        all_drone_est_expect_errors.append(DBD.fun_dist_3d(drone_dict[drone_id].xEst, drone_dict[drone_id].xTrue,
                                                           drone_dict[drone_id].D_heightEst,
                                                           drone_dict[drone_id].heightTrue))

        all_drone_height_actual.append(drone_dict[drone_id].heightActual)
        all_drone_height_exp.append(drone_dict[drone_id].heightTrue)
        all_drone_height_est.append(drone_dict[drone_id].D_heightEst)

        # ------- START: save trajectory figure ----------------
        str_step = "_XYstep_" + str(MOVE_STEP) + "cm"
        str_h_step = "_Hstep_" + str(LAYER_STEP) + "cm"
        str_cmd = str(round(normal_noise_motion_control[0] / SPEED, 2))
        str_rssi = str(rssi_std_est_target)
        if flag_known_data == 1:
            str_layer = "all"
        else:
            str_layer = str(int(current_layer * 100)) + "cm"

        drone_id = 0
        str_drone_error = str(round(all_drone_est_actual_errors[drone_id], 2))

        str_file_name = "_targets_" + str(len(list_target_beacon_IDs))

        str_file_name = str_file_name + "_layer_" + str_layer + str_step + str_h_step + "_cmd_" + str_cmd \
                        + "_rssi_" + str_rssi + "_known_data_" + str(flag_known_data) + "_anchor_" \
                        + str(FLAG_LADDER) + "_seed_" + str(tmpseed)
        if flag_known_data == 1:
            str_file_name = str_file_name + "_for_lmid_" + str(lmid_rp)

        str_est_errors = "_drone_" + str_drone_error + "_target"

        for lmid in list_target_beacon_IDs:
            str_target_error = str(round(drone_dict[drone_id].lm_est_errors[lmid][-1], 2))
            str_est_errors = str_est_errors + "_" + str(list_target_beacons[lmid][0]) \
                             + "_" + str(list_target_beacons[lmid][1]) + "_" + str_target_error

        plt.savefig(outputFolder + "Trjc" + str_file_name + str_est_errors + ".eps")

        beacon_0_pos = []
        for drone_id in range(0, num_drones):
            print("\n ********-------------------- Drone ID = %d" % drone_id)
            for lmid in range(lmid_low, lmid_high):
                if lmid in list_target_beacon_IDs:
                    print("lmid %d" % lmid)
                    for tmp_idx in range(0, len(drone_dict[drone_id].lm_est_x[lmid])):
                        print("%d, (%d, %d), %.2f " % (tmp_idx, round(drone_dict[drone_id].lm_est_x[lmid][tmp_idx]),
                                                           round(drone_dict[drone_id].lm_est_y[lmid][tmp_idx]),
                                                           drone_dict[drone_id].lm_est_errors[lmid][tmp_index]) )
                        beacon_0_pos.append((round(drone_dict[drone_id].lm_est_x[lmid][tmp_idx]),
                                             round(drone_dict[drone_id].lm_est_y[lmid][tmp_idx]),
                                             round(drone_dict[drone_id].lm_est_h[lmid][tmp_idx])))

        tmp_value, tmp_counts = np.unique(beacon_0_pos, return_counts=True, axis=0)
        # print("\n\n---------------- values, counts ------------")
        # print(tmp_counts)
        # print(tmp_value)
        tmp_entropy = entropy(tmp_counts, base=2)
        x = (np.array(beacon_0_pos)).T
        tmp_cov = np.cov(x)
        print("\n\n---------------- entropy & var/cov ------------")
        print(tmp_entropy)
        print(tmp_cov)

        GLOBAL_ENTROPY_LIST.append((flag_ladder, entropy(tmp_counts, base=2)))
        GLOBAL_VAR_LIST.append((flag_ladder, np.cov(x)))

        with open(outputFolder + 'data_entropy_var' + str_file_name + '.txt', 'w') as ftmp:
            ftmp.write("%.2f\n" % tmp_entropy)
            ftmp.write("%.2f  %.2f  %.2f\n" % (tmp_cov[0, 0], tmp_cov[0, 1], tmp_cov[1, 1]))

        with open(outputFolder + 'data_target_pos' + str_file_name + '.txt', 'w') as ftmp:
            drone_id = 0
            if lmid in list_target_beacon_IDs:
                for tmp_idx in range(0, len(drone_dict[drone_id].lm_est_x[lmid])):
                    tmpnew = ("%d  %.2f  %.2f  %.2f  %.2f" % (tmp_idx, drone_dict[drone_id].lm_est_x[lmid][tmp_idx],
                                                              drone_dict[drone_id].lm_est_y[lmid][tmp_idx],
                                                              drone_dict[drone_id].lm_est_h[lmid][tmp_idx],
                                                              drone_dict[drone_id].lm_est_errors[lmid][tmp_index]))
                    ftmp.write("%s\n" % tmpnew)

        # -------------------------------------------------------------------------
        drone_id = 0
        # with open(outputFolder + 'hxTrue' + str_file_name + '.pkl', 'wb') as f:
        #     pickle.dump(drone_dict[drone_id].hxTrue, f)
        #
        # with open(outputFolder + 'hxActual' + str_file_name + '.pkl', 'wb') as f:
        #     pickle.dump(drone_dict[drone_id].hxActual, f)
        #
        # with open(outputFolder + 'hxEst' + str_file_name + '.pkl', 'wb') as f:
        #     pickle.dump(drone_dict[drone_id].hxEst, f)

        """
        # 07/09/20: save heights
        with open( outputFolder + 'hHeightTrue'+str_file_name+'.pkl', 'wb') as f:
            pickle.dump(drone_dict[drone_id].hHeightTrue, f)
        with open( outputFolder + 'hHeightActual'+str_file_name+'.pkl', 'wb') as f:
            pickle.dump(drone_dict[drone_id].hHeightActual, f)
        with open( outputFolder + 'hHeightEst'+str_file_name+'.pkl', 'wb') as f:
            pickle.dump(drone_dict[drone_id].hHeightEst, f)
        """

        # for lmid in range(drone_dict[drone_id].lmid_low, drone_dict[drone_id].lmid_high):
        #     if flag_known_data == 1:
        #         b_id = ""
        #     else:
        #         b_id = '_target_beacon_' + str(lmid)
        #     if lmid < drone_dict[drone_id].anchor_start_lmid:
        #         with open(outputFolder + 'hxRSSI' + str_file_name + b_id + '.pkl', 'wb') as f:
        #             pickle.dump(drone_dict[drone_id].hxRSSIdict[lmid], f)
        #     elif (drone_dict[drone_id].flag_ladder == 1) and (
        #             lmid in drone_dict[drone_id].deployed_anchor_beacon_index_list):
        #         with open(outputFolder + 'hxRSSI' + str_file_name + b_id + '.pkl', 'wb') as f:
        #             pickle.dump(drone_dict[drone_id].hxRSSIdict[lmid], f)

        # ------- END: save trajectory figure ----------------

        tmpnew = ("%6.3f" % RSSI_STD_TARGET)
        tmpnew = tmpnew + ("  %6.3f" % normal_noise_motion_control[0])
        tmpnew = tmpnew + ("  %d" % FLAG_LADDER)
        tmpnew = tmpnew + ("  %6.2f" % dict_all_beacons_errors[0][0])  # beacon 0's estimation error
        tmpnew = tmpnew + ("  %d" % (len(GLOBAL_LIST_DIFF_D)))
        tmpnew = tmpnew + ("  %6.3f" % (np.mean(GLOBAL_LIST_DIFF_D)))
        tmpnew = tmpnew + ("  %6.3f" % (np.std(GLOBAL_LIST_DIFF_D)))
        with open('tmpdata.txt', 'a') as ftmp:
            tmpnew = tmpnew + ("%6.2f" % DBD.fun_dist_3d(drone_dict[drone_id].xEst, drone_dict[drone_id].xActual,
                                                         drone_dict[drone_id].D_heightEst,
                                                         drone_dict[drone_id].heightActual))
            ftmp.write("%s\n" % tmpnew)

    avg_error_drone_est_actual = 0
    avg_error_drone_est_expect = 0
    tmp = 0
    err_list_beacon_1 = []
    err_list_drone_actual = []
    err_list_drone_expect = []
    for idx in range(len(rnd_seed_list)):
        print("\n\n------------- find averages over all random seeds -------------")
        dict_all_beacons_errors = all_beacon_errors[idx]

        list_last_est_xyh = []
        print("lmid\test dist err,\tactual x,y,z\test x,y,z")
        for i in dict_all_beacons_errors.keys():
            # Get three items: distance error, beacon's actual pos (x, y), beacon's estimated pos (x, y).
            if i == 1:  # Only look at the Beacon with index 1
                tmp = tmp + dict_all_beacons_errors[i][0]
            if flag_known_data == 1:
                b_id = lmid_rp
            else:
                b_id = i
            print(b_id, "\t%6.2f" % dict_all_beacons_errors[i][0], \
                  "\t( %6.2f, %6.2f, %6.2f )" % (dict_all_beacons_errors[i][1][0], dict_all_beacons_errors[i][1][1],
                                                 dict_all_beacons_errors[i][1][2]), \
                  "\t( %6.2f, %6.2f, %6.2f )" % (dict_all_beacons_errors[i][2][0], dict_all_beacons_errors[i][2][1],
                                                 dict_all_beacons_errors[i][2][2],))
            list_last_est_xyh.append([dict_all_beacons_errors[i][2][0],
                                      dict_all_beacons_errors[i][2][1],
                                      dict_all_beacons_errors[i][2][2]])
        # save xyh
        # with open("data3d/last_Est_xyh_layer_" + str(LAYER) + ".pkl", 'wb') as f:
        #     pickle.dump(list_last_est_xyh, f)

        # err_list_beacon_1.append( dict_all_beacons_errors[1][0] ) # only look Beacon 1.
        err_list_drone_actual.append(all_drone_est_actual_errors[idx])
        err_list_drone_expect.append(all_drone_est_expect_errors[idx])

        print("drone err est from actual,   %.2f" % all_drone_est_actual_errors[idx])
        print("drone err est from expected, %.2f" % all_drone_est_expect_errors[idx])

        avg_error_drone_est_actual = avg_error_drone_est_actual + all_drone_est_actual_errors[idx]
        avg_error_drone_est_expect = avg_error_drone_est_expect + all_drone_est_expect_errors[idx]

        print("drone expect loc  (%.2f, %.2f, %.2f)" % (all_drone_locs_exp[idx][0], all_drone_locs_exp[idx][1],
                                                        all_drone_height_exp[idx]))
        print("drone actual loc  (%.2f, %.2f, %.2f)" % (all_drone_locs_actual[idx][0], all_drone_locs_actual[idx][1],
                                                        all_drone_height_actual[idx]))
        print("drone est    loc  (%.2f, %.2f, %.2f)" % (all_drone_locs_est[idx][0], all_drone_locs_est[idx][1],
                                                        all_drone_height_est[idx]))

    print("\navg drone err est from actual, \t%.2f  " % (avg_error_drone_est_actual / len(rnd_seed_list)))
    print("avg drone err est from expected, \t%.2f" % (avg_error_drone_est_expect / len(rnd_seed_list)))

    print("Drone estimation error from actual pos")
    tmpstr = ""
    for item in err_list_drone_actual:
        tmpstr = tmpstr + ("%.2f, " % item)
        # print("%.2f" % item)
    print(tmpstr)
    print("\n")

    # input("pause....wait for file saving")

    # print("\nDrone estimation error from expected pos")
    # tmpstr = ""
    # for item in err_list_drone_expect:
    #    tmpstr = tmpstr + ("%.2f, " % item)
    #    # print("%.2f" % item)
    # print(tmpstr)

    # with open('file_P_'+str(flag_ladder)+'.txt', 'w') as f:
    #    for item in GLOBAL_LIST_UD_PARTICLE:
    #        f.write("%s\n" % item)
    # with open('file_D_' + str(flag_ladder) + '.txt', 'w') as f:
    #    for item in GLOBAL_LIST_UD_DRONE:
    #        f.write("%s\n" % item)

    '''
    print("\n\n")
    for i in GLOBAL_LIST_UD_DRONE:
        print(i)
        #print("%6.2f, %6.2f" % (i[0], i[1]) )

    print("\n\n")
    for i in GLOBAL_LIST_UD_PARTICLE:
        print(i)
        #print("%6.2f, %6.2f" % (i[0], i[1]) )
    '''


def get_mid_pt_list(restart):
    if restart == 0:
        mid_pt_list = [(0, 10), (0, 20), (10, 20), (10, 30), (10, 40), (20, 40), (30, 40), (40, 40), (40, 30), (30, 30),
                       (30, 40),
                       (30, 30), (40, 30), (40, 20), (30, 20), (30, 30), (20, 30), (20, 40), (20, 30), (10, 30),
                       (0, 30), (0, 40),
                       (10, 40), (10, 30), (20, 30), (20, 20), (30, 20), (30, 10), (40, 10), (40, 20), (40, 10),
                       (40, 0), (30, 0),
                       (30, 10), (20, 10), (20, 20), (10, 20), (10, 30), (0, 30), (0, 20), (10, 20), (10, 10), (20, 10),
                       (20, 0),
                       (30, 0), (40, 0), (40, 10), (40, 0), (30, 0), (20, 0), (10, 0), (10, 10), (0, 10), (0, 0),
                       (10, 0), (0, 0)]
    else:
        mid_pt_list = []

    return mid_pt_list


if __name__ == '__main__':

    folder_names = ["data/", "data3d/", "data_replay/"]
    for name in folder_names:
        if not exists(name):
            try:
                mkdir(name)
            except OSError:
                print("failed to create directory %s" % name)
            else:
                print("created directory %s" % name)

    global_rssi_std_target = np.array([0.1])
    global_drone_translational_speed_std = np.array([0.05])
    global_rnd_seed_list = [1]

    FLAG_FASTSLAM2 = 0  # use fastSLAM2.0 = 1 or 1.0 = 0

    # fff flags -------------------------------------------------------------------
    global_FLAG_LADDER = [1]  # for determining to use anchor as ladder or not
    flag_restart = 1
    start = (0, 0, 0)

    flag_replay = 0 # always 0 here
    lmid_rp = 6  # only used in replay #s1

    #-------------------------------------------------------------------------------

    RSSI_N = 1.68
    RSSI_A0 = -47.29
    flag_graph_data = 0
    DEBUG = 1
    # show_animation = True
    show_animation = False

    MOVE_STEP = 100  # in cm
    DT = 1
    SPEED = MOVE_STEP / 100.0 / DT
    SCALE_MOTION_STD_DRONE = 1

    # up
    UP_STEP = 100  # cm
    SPEED_UP = UP_STEP / 100.0 / DT
    LAYER_STEP = UP_STEP
    height_std = 0.03  # m
    up_var = 0.03 ** 2  #

    # 10x10x3 room
    instance_max_coord_x = 40  # 40
    instance_max_coord_y = 40  # 40
    instance_max_coord_h = 3  # 3

    # initiall guess of beacon poistion (x, y)'s var-covar matrix P, this is different from R.
    BEACON_COV_GUESS = np.diag([(1 * MOVE_STEP / 100.0) ** 2, (1 * MOVE_STEP / 100.0) ** 2,
                                (1 * MOVE_STEP / 100.0) ** 2])  # 07/07/20: 3x3, 3d
    BEACON_COV_GUESS_ANCHOR = 0.001 * BEACON_COV_GUESS  #

    # target beacons' actual locations.
    # list_target_beacons = [[2, 3, 0], [5, 5, 0], [7, 2, 1], [5, 5, 1], [3, 8, 2], [8, 6, 3]] # 10x10x3
    # list_target_beacons = [[5, 5, 0], [10, 10, 0], [16, 3, 1], [10, 10, 2], [4, 18, 2], [20, 15, 3]] # 20x20x3
    # list_target_beacons = [[20, 0, 0], [40, 0, 0], [40, 20, 0.5], [40, 40, 1], [20, 40, 1], [0, 40, 1.5], [0, 20, 1.5],
    #                     [10, 10, 1.75], [30, 10, 2], [30, 30, 2.5], [10, 30, 2.75]] # 40x40x3, 11 targets
    list_target_beacons_all = [[1, 12, 1.5], [11, 7, 2.5], [40, 0, 0], [31, 14, 0.5], [20, 20, 3], [6, 25, 0.75],
                           [40, 25, 1],
                           [27, 30, 0.25], [2, 39, 2], [23, 40, 1.75]]  # 40x40x3, 10 targets
    list_target_beacons = [list_target_beacons_all[lmid_rp],list_target_beacons_all[lmid_rp],list_target_beacons_all[lmid_rp],
                           list_target_beacons_all[lmid_rp],list_target_beacons_all[lmid_rp],list_target_beacons_all[lmid_rp],
                           list_target_beacons_all[lmid_rp],list_target_beacons_all[lmid_rp],list_target_beacons_all[lmid_rp],
                           list_target_beacons_all[lmid_rp]]

    list_target_beacons_xy = list(l[0:2] for l in list_target_beacons)
    list_target_beacons_h = list(l[2] for l in list_target_beacons)

    list_anchor_beacons = [[0, 0, 0]]
    list_anchor_beacons_xy = list(l[0:2] for l in list_anchor_beacons)
    list_anchor_beacons_h = list(l[2] for l in list_anchor_beacons)

    if flag_restart == 0:
        start = (0, 0, 0)
        old_deployed_anchors_index = []
        old_deployed_anchors_actual = {}
        old_deployed_anchors_est = {}

        dict_xTrue_dict1 = {}
    else:
        with open('dict_xTrue_dict1.pk', 'rb') as f:
            dict_xTrue_dict1 = pickle.load(f)

        # read data of last stage from files================================================
        with open('deployed_anchor_index.pk', 'rb') as f:
            old_deployed_anchors_index = pickle.load(f)

        with open('deployed_anchor_act.pk', 'rb') as f:
            old_deployed_anchors_actual = pickle.load(f)

        with open('deployed_anchor_est.pk', 'rb') as f:
            old_deployed_anchors_est = pickle.load(f)

    # mid_pt_list = [(5, 5), (5, 10), (10, 10), (10, 5), (5, 5), (0, 5), (0, 0)] # 10x10
    # mid_pt_list = [(20, 20), (20, 40), (40, 40), (40, 20), (20, 20), (0, 20), (0, 0)] # 40x40
    # mid_pt_list = [(10, 10), (10, 20), (20, 20), (20, 10), (10, 10), (0, 10), (0, 0)] # 20x20
    mid_pt_list = get_mid_pt_list(flag_restart)
    beacon_loc_guess_array = [[0, 0, 0]]

    mid_pt_list_3d = []

    for tmp_rssi_std in global_rssi_std_target:
        for tmp_speed_std in global_drone_translational_speed_std:
            for flag_ladder in global_FLAG_LADDER:
                for tmpseed in global_rnd_seed_list:

                    GLOBAL_LIST_D_RSSI = []
                    GLOBAL_LIST_DIFF_D = []

                    rnd_seed_list = [tmpseed]
                    normal_noise_motion_control = np.array([tmp_speed_std * SPEED, 0.00001 * 3.14 / 180, 0,
                                                            height_std * SPEED_UP, 0.00001 * 3.14 / 180, 0])

                    R_TARGET = 0.0001  # variance of target beacons' measurement data in their EKFs' calculations.
                    R_ANCHOR = 0.0001  # variance of anchor beacons' measurement data in their EKFs' calculations.
                    RSSI_STD_TARGET = tmp_rssi_std  # this adds noise to real RSSI data of target beacons
                    RSSI_STD_ANCHOR = np.sqrt(R_ANCHOR * 0.0001)  # this adds noise to real RSSI data of anchor beacons
                    scale_rssi_noise = 1
                    scale_rssi_noise_anchors = 1

                    # --------------------------------- ---------------------------------

                    if flag_replay == 0:
                        flag_known_data = 0
                        FLAG_KNOWN_DATA = flag_known_data
                        outputFolder = "data/"
                        pklxTrue = ""
                        pklRSSI = ""
                        # 07/02/2020: modified for 3d
                        # for layer in range(0, instance_max_coord_h, up_step):
                        # changed list_target/anchor_beacons(s) to *_xy(s)
                        # added layer, instance_max_coord_h, list_target/anchor_beacons_h(s),
                        # height_std, up_var,
                        # list_target/anchor_beacons has [x,y,z], different from those in Drone()

                        # 07/11/20: the for loop does not added lm for next layer correctly, need to fix later
                        # for layer in range(0, instance_max_coord_h * 100 + 1, UP_STEP): #replay
                        #     LAYER = int(layer)  # cm, Global
                        #     current_layer = float(layer / 100.)  # m
                        #     print("going to run SLAM on layer at %.2f m" % current_layer)
                        #     # input("pause...")
                        #
                        #     mid_pt_list_3d = []
                        #     for i in range(len(mid_pt_list)):
                        #         mid_pt_list_3d.append((mid_pt_list[i][0], mid_pt_list[i][1], int(current_layer)))
                        #     print(mid_pt_list_3d)
                        #     # input("pause")
                        #     # mid_pt_list_3d = [(20, 20, 2), (20, 40, 1), (40, 40, 3), (40, 20, 0), (20, 20, 1), (0, 20, 1), (0, 0, 0)]  # 40x40

                        FastSLAM_Run(RSSI_N, RSSI_A0, rnd_seed_list, flag_graph_data, flag_ladder, DEBUG,
                                     show_animation,
                                     R_TARGET, R_ANCHOR, RSSI_STD_TARGET, RSSI_STD_ANCHOR,
                                     scale_rssi_noise, scale_rssi_noise_anchors,
                                     MOVE_STEP, UP_STEP, DT, SPEED, normal_noise_motion_control,
                                     SCALE_MOTION_STD_DRONE, height_std, up_var,
                                     instance_max_coord_x, instance_max_coord_y, instance_max_coord_h,
                                     0, start,
                                     list_target_beacons, list_target_beacons_xy, list_target_beacons_h,
                                     list_anchor_beacons, list_anchor_beacons_xy, list_anchor_beacons_h,
                                     mid_pt_list_3d, beacon_loc_guess_array, BEACON_COV_GUESS,
                                     BEACON_COV_GUESS_ANCHOR,
                                     flag_known_data, pklxTrue, pklRSSI, outputFolder, lmid_rp)

                    # ---------------------------------replay---------------------------------

                    if flag_replay == 1:
                        flag_known_data = 1
                        FLAG_KNOWN_DATA = flag_known_data
                        FLAG_EST_INIT = 1  # 1=use last est as init for replay, 0=not use
                        if FLAG_EST_INIT == 1:
                            with open("data3d/last_Est_xyh_layer_" + str(int(instance_max_coord_h * 100)) + ".pkl",
                                      'rb') as f:
                                last_Est_xyh = pickle.load(f)

                        inputFolder = "data3d/"
                        outputFolder = "data_replay/"

                        num_target = len(list_target_beacons)  # total number of targets

                        # merging pkl files in data/ folder and output them to data3d/ folder
                        CP.combine_pkl_files(num_target, UP_STEP, int(instance_max_coord_h * 100), MOVE_STEP)

                        # lmid_rp = 0 # for single target used
                        str_filename = str(num_target) + "_layers_all_XYstep_" + str(MOVE_STEP) + "cm_Hstep_" \
                                       + str(LAYER_STEP) + "cm_cmd_0.1_rssi_0.1_known_data_0_anchor_0_seed_1"

                        # ----------- pklxTrue and pklRSSI_List are valid only when flag_known_data=1.
                        ##------all RSSI use------
                        pklxTrue = inputFolder + "hxTrue_targets_" + str_filename + ".pkl"
                        ##------specific RSSI use -------
                        # layer_selected = 250 #cm
                        # pklxTrue = "data/hxTrue_targets_11_layer_" + str(layer_selected) + "cm_XYstep_100cm_Hstep_50cm_cmd_0.1_rssi_0.1_known_data_0_anchor_0_seed_1_up_or_down_1.pkl"
                        # pklhTrue = "data/hHeightTrue_targets_11_layer_" + str(layer_selected) + "cm_XYstep_100cm_Hstep_50cm_cmd_0.1_rssi_0.1_known_data_0_anchor_0_seed_1_up_or_down_1.pkl"

                        for lmid_rp in range(0, num_target):  # all
                            GL_LIST_LAST_B_EST[0] = last_Est_xyh[lmid_rp]
                            print("last est: ", GL_LIST_LAST_B_EST)
                            # input(",,,pause,,,")
                            # LMID = 10 # single
                            # for lmid_rp in range(LMID, LMID+1): # single
                            # all RSSI
                            pklRSSI = inputFolder + "hxRSSI_targets_" + str_filename + "_target_beacon_" + str(
                                lmid_rp) + ".pkl"
                            # specific RSSI
                            # pklRSSI = "data/hxRSSI_targets_11_layer_" + str(layer_selected) \
                            #          + "cm_XYstep_100cm_Hstep_50cm_cmd_0.1_rssi_0.1_known_data_0_anchor_0_seed_1_up_or_down_1_target_beacon_" \
                            #          + str(lmid_rp) + ".pkl"
                            single_one_in_list_target_beacons = [
                                list_target_beacons[lmid_rp]]  # during replay, we only choose one target a time.
                            single_one_in_list_target_beacons_xy = list(
                                l[0:2] for l in single_one_in_list_target_beacons)
                            single_one_in_list_target_beacons_h = list(l[2] for l in single_one_in_list_target_beacons)
                            print("going to replay SLAM to more accurately localize a target beacon",
                                  "single_one_in_list_target_beacons -->>", single_one_in_list_target_beacons)
                            input("...pause...")

                            FastSLAM_Run(RSSI_N, RSSI_A0, rnd_seed_list, flag_graph_data, flag_ladder, DEBUG,
                                         show_animation,
                                         R_TARGET, R_ANCHOR, RSSI_STD_TARGET, RSSI_STD_ANCHOR,
                                         scale_rssi_noise, scale_rssi_noise_anchors,
                                         MOVE_STEP, UP_STEP, DT, SPEED, normal_noise_motion_control,
                                         SCALE_MOTION_STD_DRONE, height_std, up_var,
                                         instance_max_coord_x, instance_max_coord_y, instance_max_coord_h, 0, start,
                                         single_one_in_list_target_beacons, single_one_in_list_target_beacons_xy,
                                         single_one_in_list_target_beacons_h,
                                         list_anchor_beacons, list_anchor_beacons_xy, list_anchor_beacons_h,
                                         mid_pt_list_3d, beacon_loc_guess_array, BEACON_COV_GUESS,
                                         BEACON_COV_GUESS_ANCHOR,
                                         flag_known_data, pklxTrue, pklRSSI, outputFolder, lmid_rp)

                # main()
    # print(sorted(GLOBAL_LIST))
    # plt.figure("GLOBAL LIST")
    # plt.plot(GLOBAL_LIST, 'rx-')

    # check for up to date 07/24 00:10

    # plt.show()