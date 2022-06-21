import pickle
import numpy as np
import math
from os.path import exists

# ver 2.2
# renamed directories
# ---------------------------------------------------------------------------------\

inputFolder = "data/"
outputFolder = "data3d/"


# ---------------------------------------------------------------------------------

def add_xTrue_down(num_target, up_step, up_lim, move_step):
    pkl_xTrue_all = np.array([[], [], []])
    str_filename = "hxTrue_targets_" + str(num_target) + "_layer_" + str(up_lim) \
                   + "cm_XYstep_" + str(move_step) + "cm_Hstep_" + str(up_step) \
                   + "cm_cmd_0.1_rssi_0.1_known_data_0_anchor_0_seed_1_up_or_down_0"
    if exists(inputFolder + str_filename + ".pkl"):
        with open(inputFolder + str_filename + ".pkl", 'rb') as f:
            pkl_xTrue = pickle.load(f)
        pkl_xTrue_all = np.hstack((pkl_xTrue_all, pkl_xTrue))
    for layer in range(up_lim-up_step, -1, -up_step):
        str_filename = "hxTrue_targets_" + str(num_target) + "_layer_" + str(layer) \
                   + "cm_XYstep_" + str(move_step) + "cm_Hstep_" + str(up_step) \
                   + "cm_cmd_0.1_rssi_0.1_known_data_0_anchor_0_seed_1_up_or_down_-1"
        with open(inputFolder + str_filename + ".pkl", 'rb') as f:
            pkl_xTrue = pickle.load(f)
        pkl_xTrue_all = np.hstack((pkl_xTrue_all, pkl_xTrue))
    return pkl_xTrue_all


def add_hTrue_down(num_target, up_step, up_lim, move_step):
    pkl_hTrue_all = np.array([])
    str_filename = "hHeightTrue_targets_" + str(num_target) + "_layer_" + str(up_lim) \
                   + "cm_XYstep_" + str(move_step) + "cm_Hstep_" + str(up_step) \
                   + "cm_cmd_0.1_rssi_0.1_known_data_0_anchor_0_seed_1_up_or_down_0"
    if exists(inputFolder + str_filename + ".pkl"):
        with open(inputFolder + str_filename + ".pkl", 'rb') as f:
            pkl_hTrue = pickle.load(f)
        pkl_hTrue_all = np.hstack((pkl_hTrue_all, pkl_hTrue))

    for layer in range(up_lim-up_step, -1, -up_step):
        str_filename = "hHeightTrue_targets_" + str(num_target) + "_layer_" + str(layer) \
                       + "cm_XYstep_" + str(move_step) + "cm_Hstep_" + str(up_step) \
                       + "cm_cmd_0.1_rssi_0.1_known_data_0_anchor_0_seed_1_up_or_down_-1"
        with open(inputFolder + str_filename + ".pkl", 'rb') as f:
            pkl_hTrue = pickle.load(f)
        pkl_hTrue_all = np.hstack((pkl_hTrue_all, pkl_hTrue))

    return pkl_hTrue_all


def add_xRSSI_down(lmid, num_target, up_step, up_lim, move_step):
    pkl_xRSSI_all = np.array([])
    str_filename = "hxRSSI_targets_" + str(num_target) + "_layer_" + str(up_lim) \
                   + "cm_XYstep_" + str(move_step) + "cm_Hstep_" + str(up_step) \
                   + "cm_cmd_0.1_rssi_0.1_known_data_0_anchor_0_seed_1_up_or_down_0_target_beacon_" \
                   + str(lmid)
    if exists(inputFolder + str_filename + ".pkl"):
        with open(inputFolder + str_filename + ".pkl", 'rb') as f:
            pkl_xRSSI = pickle.load(f)
        pkl_xRSSI_all = np.hstack((pkl_xRSSI_all, pkl_xRSSI))
    for layer in range(up_lim-up_step, -1, -up_step):
        str_filename = "hxRSSI_targets_" + str(num_target) + "_layer_" + str(layer) \
                       + "cm_XYstep_" + str(move_step) + "cm_Hstep_" + str(up_step) \
                       + "cm_cmd_0.1_rssi_0.1_known_data_0_anchor_0_seed_1_up_or_down_-1_target_beacon_" \
                       + str(lmid)
        with open(inputFolder + str_filename + ".pkl", 'rb') as f:
            pkl_xRSSI = pickle.load(f)
        pkl_xRSSI_all = np.hstack((pkl_xRSSI_all, pkl_xRSSI))
    return pkl_xRSSI_all

# ---------------------------------------------------------------------------------

def combine_pkl_files(include_down, num_target, up_step, up_lim, move_step):

    # save xTrue at all layers
    pkl_xTrue_all = np.array([[],[],[]]) # x,y,yaw
    for layer in range(0,up_lim+1, up_step):
        #print(layer)
        str_filename = "hxTrue_targets_" + str(num_target) + "_layer_" + str(layer) \
                       + "cm_XYstep_" + str(move_step) + "cm_Hstep_" + str(up_step) \
                       + "cm_cmd_0.1_rssi_0.1_known_data_0_anchor_0_seed_1_up_or_down_1"
        with open(inputFolder + str_filename + ".pkl", 'rb') as f:
            pkl_xTrue = pickle.load(f)
        pkl_xTrue_all = np.hstack((pkl_xTrue_all, pkl_xTrue))

    if include_down == 1:
        pkl_xTrue_all = np.hstack((pkl_xTrue_all, add_xTrue_down(num_target, up_step, up_lim, move_step)))

    str_filename = "hxTrue_targets_" + str(num_target) \
                    + "_layers_all_XYstep_" + str(move_step) + "cm_Hstep_" + str(up_step) \
                   + "cm_cmd_0.1_rssi_0.1_known_data_0_anchor_0_seed_1"
    with open(outputFolder + str_filename + ".pkl", 'wb') as f:
        pickle.dump(pkl_xTrue_all, f)
    #print(len(pkl_xTrue_all[0]))
    #print(pkl_xTrue_all[0])


    # save hTrue at all layers
    pkl_hTrue_all = np.array([]) # h
    for layer in range(0, up_lim + 1, up_step):
        str_filename = "hHeightTrue_targets_" + str(num_target) + "_layer_" + str(layer) \
                       + "cm_XYstep_" + str(move_step) + "cm_Hstep_" + str(up_step) \
                       + "cm_cmd_0.1_rssi_0.1_known_data_0_anchor_0_seed_1_up_or_down_1"
        with open(inputFolder + str_filename + ".pkl", 'rb') as f:
            pkl_hTrue = pickle.load(f)
        pkl_hTrue_all = np.hstack((pkl_hTrue_all, pkl_hTrue))
    #print(pkl_hTrue_all)

    if include_down == 1:
        pkl_hTrue_all = np.hstack((pkl_hTrue_all, add_hTrue_down(num_target, up_step, up_lim, move_step)))

    str_filename = "hHeightTrue_targets_" + str(num_target) \
                    + "_layers_all_XYstep_" + str(move_step) + "cm_Hstep_" + str(up_step) \
                   + "cm_cmd_0.1_rssi_0.1_known_data_0_anchor_0_seed_1"
    with open(outputFolder + str_filename + ".pkl", 'wb') as f:
        pickle.dump(pkl_hTrue_all, f)
    #print(len(pkl_hTrue_all))
    #print(pkl_hTrue_all)


    # save RSSI for each target at all layers
    for lmid in range(num_target):
        pkl_xRSSI_all = np.array([])  # RSSI
        for layer in range(0, up_lim + 1, up_step):
            str_filename = "hxRSSI_targets_" + str(num_target) + "_layer_" + str(layer) \
                           + "cm_XYstep_" + str(move_step) + "cm_Hstep_" + str(up_step) \
                           + "cm_cmd_0.1_rssi_0.1_known_data_0_anchor_0_seed_1_up_or_down_1_target_beacon_" \
                        + str(lmid)
            with open(inputFolder + str_filename + ".pkl", 'rb') as f:
                pkl_xRSSI = pickle.load(f)
            pkl_xRSSI_all = np.hstack((pkl_xRSSI_all, pkl_xRSSI))

        if include_down == 1:
            pkl_xRSSI_all = np.hstack((pkl_xRSSI_all, add_xRSSI_down(lmid, num_target, up_step, up_lim, move_step)))

        str_filename = "hxRSSI_targets_" + str(num_target) \
                    + "_layers_all_XYstep_" + str(move_step) + "cm_Hstep_" + str(up_step) \
                    + "cm_cmd_0.1_rssi_0.1_known_data_0_anchor_0_seed_1_target_beacon_" \
                    + str(lmid)
        with open(outputFolder + str_filename + ".pkl", 'wb') as f:
            pickle.dump(pkl_xRSSI_all, f)
        #print(len(pkl_xRSSI_all))
        #print(pkl_xRSSI_all)

# ---------------------------------------------------------------------------------
