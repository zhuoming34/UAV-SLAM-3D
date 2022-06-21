import pickle
import numpy as np
import math
from os.path import exists

# 0723
# merged true height and pitch into xTrue
# ---------------------------------------------------------------------------------\

inputFolder = "data/"
outputFolder = "data3d/"

# ---------------------------------------------------------------------------------

def combine_pkl_files(num_target, up_step, up_lim, move_step):

    # save xTrue at all layers
    pkl_xTrue_all = np.array([[],[],[],[],[]]) # x,y,yaw
    for layer in range(0,up_lim+1, up_step):
        #print(layer)
        str_filename = "hxTrue_targets_" + str(num_target) + "_layer_" + str(layer) \
                       + "cm_XYstep_" + str(move_step) + "cm_Hstep_" + str(up_step) \
                       + "cm_cmd_0.1_rssi_0.1_known_data_0_anchor_0_seed_1"
        with open(inputFolder + str_filename + ".pkl", 'rb') as f:
            pkl_xTrue = pickle.load(f)
        pkl_xTrue_all = np.hstack((pkl_xTrue_all, pkl_xTrue))

    str_filename = "hxTrue_targets_" + str(num_target) \
                    + "_layers_all_XYstep_" + str(move_step) + "cm_Hstep_" + str(up_step) \
                   + "cm_cmd_0.1_rssi_0.1_known_data_0_anchor_0_seed_1"
    with open(outputFolder + str_filename + ".pkl", 'wb') as f:
        pickle.dump(pkl_xTrue_all, f)
    #print(len(pkl_xTrue_all[0]))
    #print(pkl_xTrue_all[0])

    # save RSSI for each target at all layers
    for lmid in range(num_target):
        pkl_xRSSI_all = np.array([])  # RSSI
        for layer in range(0, up_lim + 1, up_step):
            str_filename = "hxRSSI_targets_" + str(num_target) + "_layer_" + str(layer) \
                           + "cm_XYstep_" + str(move_step) + "cm_Hstep_" + str(up_step) \
                           + "cm_cmd_0.1_rssi_0.1_known_data_0_anchor_0_seed_1_target_beacon_" \
                        + str(lmid)
            with open(inputFolder + str_filename + ".pkl", 'rb') as f:
                pkl_xRSSI = pickle.load(f)
            pkl_xRSSI_all = np.hstack((pkl_xRSSI_all, pkl_xRSSI))

        str_filename = "hxRSSI_targets_" + str(num_target) \
                    + "_layers_all_XYstep_" + str(move_step) + "cm_Hstep_" + str(up_step) \
                    + "cm_cmd_0.1_rssi_0.1_known_data_0_anchor_0_seed_1_target_beacon_" \
                    + str(lmid)
        with open(outputFolder + str_filename + ".pkl", 'wb') as f:
            pickle.dump(pkl_xRSSI_all, f)
        #print(len(pkl_xRSSI_all))
        #print(pkl_xRSSI_all)

# ---------------------------------------------------------------------------------
