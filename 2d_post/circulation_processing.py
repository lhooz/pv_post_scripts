"""main script for circulation processing"""

import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from cprocessing_functions import (field_plot, vortices_processing,
                                   vortices_tracking)

window = [-6, 4, -4, 4]
resolution = [500, 400]
threshold_q = 0.002
threshold_vorz = 0.02
threshold_circulation = 0.02
wbound_radius = 0

data_time_increment = 2e-2
v_vanish_dist_factor = 2

cwd = os.getcwd()
vorz_folder = os.path.join(cwd, 'vorz_data')
q_folder = os.path.join(cwd, 'q_data')
wgeo_folder = os.path.join(cwd, 'wgeo_data')

oimage_folder = os.path.join(cwd, 'oimages')
if os.path.exists(oimage_folder):
    shutil.rmtree(oimage_folder)
os.mkdir(oimage_folder)
#----------------------------------------
wgeo_boundx_history = []
vortices_history = []
# for ti in range(1, len(os.listdir(q_folder))):
for ti in range(1, 10):
    time_instance = str(ti).zfill(4)

    vorz_data_file = os.path.join(vorz_folder,
                                  'vorz_' + time_instance + '.csv')
    q_data_file = os.path.join(q_folder, 'q_' + time_instance + '.csv')
    wgeo_data_file = os.path.join(wgeo_folder,
                                  'wgeo_' + time_instance + '.csv')

    oimage_file = os.path.join(oimage_folder,
                               'anime_' + time_instance + '.png')

    files_dir = [vorz_data_file, q_data_file, wgeo_data_file]
    res_parameters = [window, resolution]
    threshold_parameters = [
        threshold_q, threshold_vorz, threshold_circulation, wbound_radius
    ]

    vz_circulations, image_vortices, vz_field, wgeo_bound_xi = vortices_processing(
        files_dir, res_parameters, threshold_parameters)
    # ----------------------------------------------
    wgeo_boundx_history.append(wgeo_bound_xi)
    vortices_history.append(vz_circulations)

    field_plot(window, vz_field[1], image_vortices, oimage_file, 'save')
    plt.close()
# ---------------------------------------
# print(vortices_history)

vortices_tracking(data_time_increment, wgeo_boundx_history,
                  v_vanish_dist_factor, vortices_history)
# v1 = [str(timei), str(v1[0]), str(v1[1]), str(v1[2])]
# v1 = ','.join(v1)
# with open('v1', 'a') as f:
# f.write("%s\n" % v1)
