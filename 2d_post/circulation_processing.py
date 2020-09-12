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
vortex_no_to_save_as_image = []  # enpty means not saving any vortex image
data_time_increment = 2e-3
#------------------------
wbound_radius = 0
#----q and vorz thresholds are absolute values, circulation is wrt. maximum (positive and negative)-------
threshold_q = 1e-4
threshold_vorz = 1e-2
threshold_circulation = 1e-1
#--multiply by wing displacement to determine the max. travel dist in one t_step for vortices, vortices are considered as the same one if its traveled dist is less than this value--
v_vanish_dist_factor = 1.5
#---------------------------
cwd = os.getcwd()
vorz_folder = os.path.join(cwd, 'vorz_data')
q_folder = os.path.join(cwd, 'q_data')
wgeo_folder = os.path.join(cwd, 'wgeo_data')

individual_vortex_folder = os.path.join(cwd, 'individual_vortex_history')
origin_ref_folder = os.path.join(individual_vortex_folder, 'origin_ref_folder')
oimage_folder = os.path.join(cwd, 'oimages')
#----------------------------------------
if os.path.exists(individual_vortex_folder):
    shutil.rmtree(individual_vortex_folder)
if os.path.exists(oimage_folder):
    shutil.rmtree(oimage_folder)
os.mkdir(individual_vortex_folder)
os.mkdir(oimage_folder)
os.mkdir(origin_ref_folder)
origin_ref_file = os.path.join(origin_ref_folder, 'origin_ref')
#----------------------------------------
wgeo_boundx_history = []
no_of_vortices = 0
marked_pvortices_history = []
marked_nvortices_history = []
marked_vortices_history = []

time_series_names = [f.name for f in os.scandir(q_folder) if f.is_file()]
time_series = [x.split('_')[-1] for x in time_series_names]
time_series = [int(x.split('.')[0]) for x in time_series]
start_t = np.min(np.array(time_series))
end_t = np.max(np.array(time_series))

for ti in range(start_t, end_t + 1):
    # for ti in range(24, 25):
    time_instance = str(ti).zfill(4)
    timei = (ti + 1) * data_time_increment
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
    vz_circulations, image_vortices, vz_field, wgeo_bound_xi, w_centroid = vortices_processing(
        files_dir, res_parameters, threshold_parameters)
    #-----write wing centroid history-----
    origin_ref_folder = os.path.join(individual_vortex_folder,
                                     'origin_ref_folder')
    if not os.path.exists(origin_ref_folder):
        os.mkdir(origin_ref_folder)

    centroid_ti = [str(timei), str(w_centroid[0]), str(w_centroid[1])]

    with open(origin_ref_file, 'a') as f:
        f.write("%s\n" % ', '.join(centroid_ti))

    pvortices = vz_circulations[0]
    nvortices = vz_circulations[1]
    # print(pvortices)
    vortices = np.append(pvortices, nvortices, axis=0)

    org_vorz_field = vz_field[0]
    vz_field_flags = vz_field[1]
    # -------------plot all vortices-----------
    field_plot(window, org_vorz_field, vz_field_flags, oimage_file, 'save')
    plt.close()
    # ----------------------------------------------
    wgeo_boundx_history.append(wgeo_bound_xi)
    # -------------vortices tracking/marking-------
    #-------------------------------------------------------------------
    #------tracking positive and negative vortices separately--------
    print(f'Current Time: {timei}')
    no_of_vortices, marked_pvortices_history, v_vanish_dist = vortices_tracking(
        no_of_vortices, timei, wgeo_boundx_history, v_vanish_dist_factor,
        marked_pvortices_history, pvortices, 'positive')
    no_of_vortices, marked_nvortices_history, v_vanish_dist = vortices_tracking(
        no_of_vortices, timei, wgeo_boundx_history, v_vanish_dist_factor,
        marked_nvortices_history, nvortices, 'negative')
    # print(marked_pvortices_history)
    print(f'Total No of vortices in history: {no_of_vortices}')
    print(f'Vortices identification distance: {v_vanish_dist} \n')
    #-------------write history of every individual vortex----------
    write_individual_vortex(window, time_instance, marked_pvortices_history,
                            org_vorz_field, vz_field_flags,
                            individual_vortex_folder,
                            vortex_no_to_save_as_image, w_centroid)
    write_individual_vortex(window, time_instance, marked_nvortices_history,
                            org_vorz_field, vz_field_flags,
                            individual_vortex_folder,
                            vortex_no_to_save_as_image, w_centroid)
    # -----tracking positive and negative vortices at the same time---
    # no_of_vortices, marked_vortices_history = vortices_tracking(
    # no_of_vortices, timei, wgeo_boundx_history, v_vanish_dist_factor,
    # marked_vortices_history, vortices, 'all')
    # print(f'Total No of vortices in history: {no_of_vortices} \n')
    # # -------------write history of every individual vortex----------
    # write_individual_vortex(window, time_instance, marked_vortices_history,
    # org_vorz_field, vz_field_flags,
    # individual_vortex_folder,
    # vortex_no_to_save_as_image, w_centroid)
    #---------------------------------------------------------------------
