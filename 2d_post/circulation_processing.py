"""main script for circulation processing"""

import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from cprocessing_functions import (field_plot, vortices_processing,
                                   vortices_tracking, write_individual_vortex)

#--------------input parameters------------
window = [-6, 4, -4, 4]
resolution = [1000, 800]
vortex_no_to_save_as_image = [1, 12]  # enpty means not saving any vortex image
data_time_increment = 2e-2
#------------------------
wbound_radius = 0
#----thresholds are all wrt. maximum (positive and negative)-------
threshold_q = 1e-4
threshold_vorz = 1e-3
threshold_circulation = 1e-2
#--multiply by wing displacement to determine the max. travel dist in one t_step for vortices, vortices are considered as the same one if its traveled dist is less than this value--
v_vanish_dist_factor = 1.2
#---------------------------
cwd = os.getcwd()
vorz_folder = os.path.join(cwd, 'vorz_data')
q_folder = os.path.join(cwd, 'q_data')
wgeo_folder = os.path.join(cwd, 'wgeo_data')

individual_vortex_folder = os.path.join(cwd, 'individual_vortex_history')
oimage_folder = os.path.join(cwd, 'oimages')
#----------------------------------------
if os.path.exists(individual_vortex_folder):
    shutil.rmtree(individual_vortex_folder)
if os.path.exists(oimage_folder):
    shutil.rmtree(oimage_folder)
os.mkdir(individual_vortex_folder)
os.mkdir(oimage_folder)
#----------------------------------------
wgeo_boundx_history = []
no_of_vortices = 0
marked_pvortices_history = []
marked_nvortices_history = []
marked_vortices_history = []
for ti in range(1, len(os.listdir(q_folder))):
    # for ti in range(24, 25):
    time_instance = str(ti).zfill(4)
    #--------setting up file dirs-----------
    vorz_data_file = os.path.join(vorz_folder,
                                  'vorz_' + time_instance + '.csv')
    q_data_file = os.path.join(q_folder, 'q_' + time_instance + '.csv')
    wgeo_data_file = os.path.join(wgeo_folder,
                                  'wgeo_' + time_instance + '.csv')

    oimage_file = os.path.join(oimage_folder,
                               'anime_' + time_instance + '.png')
    files_dir = [vorz_data_file, q_data_file, wgeo_data_file]
    #----------organizing parameters--------
    res_parameters = [window, resolution]
    threshold_parameters = [
        threshold_q, threshold_vorz, threshold_circulation, wbound_radius
    ]
    #----------vortices identification by field filtering----------
    vz_circulations, image_vortices, vz_field, wgeo_bound_xi = vortices_processing(
        files_dir, res_parameters, threshold_parameters)
    pvortices = vz_circulations[0]
    nvortices = vz_circulations[1]
    vortices = np.append(pvortices, nvortices, axis=0)
    # print(vortices)
    org_vorz_field = vz_field[0]
    vz_field_flags = vz_field[1]
    # -------------plot all vortices-----------
    field_plot(window, org_vorz_field, vz_field_flags, oimage_file, 'save')
    plt.close()
    # ----------------------------------------------
    wgeo_boundx_history.append(wgeo_bound_xi)
    # -------------vortices tracking/marking-------
    #-------------------------------------------------------------------
    timei = ti * data_time_increment
    #------tracking positive and negative vortices separately--------
    no_of_vortices, marked_pvortices_history = vortices_tracking(
        no_of_vortices, timei, wgeo_boundx_history, v_vanish_dist_factor,
        marked_pvortices_history, pvortices, 'positive')
    no_of_vortices, marked_nvortices_history = vortices_tracking(
        no_of_vortices, timei, wgeo_boundx_history, v_vanish_dist_factor,
        marked_nvortices_history, nvortices, 'negative')
    # print(marked_pvortices_history)
    print(f'Total No of vortices in history: {no_of_vortices} \n')
    #-------------write history of every individual vortex----------
    write_individual_vortex(window, time_instance, marked_pvortices_history,
                            org_vorz_field, vz_field_flags,
                            individual_vortex_folder,
                            vortex_no_to_save_as_image)
    write_individual_vortex(window, time_instance, marked_nvortices_history,
                            org_vorz_field, vz_field_flags,
                            individual_vortex_folder,
                            vortex_no_to_save_as_image)
    # -----tracking positive and negative vortices at the same time---
    # no_of_vortices, marked_vortices_history = vortices_tracking(
    # no_of_vortices, timei, wgeo_boundx_history, v_vanish_dist_factor,
    # marked_vortices_history, vortices, 'all')
    # print(f'Total No of vortices in history: {no_of_vortices} \n')
    # # -------------write history of every individual vortex----------
    # write_individual_vortex(window, time_instance, marked_vortices_history,
    # org_vorz_field, vz_field_flags,
    # individual_vortex_folder,
    # vortex_no_to_save_as_image)
    #---------------------------------------------------------------------
