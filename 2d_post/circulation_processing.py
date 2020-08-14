"""main script for circulation processing"""

import os

import numpy as np
from scipy import ndimage
from cprocessing_functions import field_plot, vortices_processing

window = [-6, 4, -4, 4]
resolution = [500, 400]
threshold_q = 0.002
threshold_vorz = 0.02
threshold_circulation = 0.02
wbound_radius = 0

data_time_increment = 2e-2

cwd = os.getcwd()
vorz_folder = os.path.join(cwd, 'vorz_data')
q_folder = os.path.join(cwd, 'q_data')
wgeo_folder = os.path.join(cwd, 'wgeo_data')

oimage_folder = os.path.join(cwd, 'oimages')
if os.path.exists(oimage_folder):
    shutil.rmtree(oimage_folder)
os.mkdir(oimage_folder)
#----------------------------------------
vortex_0 = '0,0,0,0'
with open('mpv', 'w') as f:
    f.write("%s\n" % vortex_0)
with open('npv', 'w') as f:
    f.write("%s\n" % vortex_0)
#----------------------------------------
mp_vortex = [[0, 0, 0, 0]]
np_vortex = [[0, 0, 0, 0]]
for ti in range(1, len(os.listdir(q_folder))):
    # for ti in range(1):
    time_instance = str(ti).zfill(4)
    timei = ti * data_time_increment

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

    sorted_vortices, image_vortices, vz_field_filtered = vortices_processing(
        files_dir, res_parameters, threshold_parameters)

    pvi = sorted_vortices[0][0]
    nvi = sorted_vortices[1][0]
    mp_vortex.append([timei, pvi[0], pvi[1], pvi[2]])
    np_vortex.append([timei, nvi[0], nvi[1], nvi[2]])

    # print(sorted_vortices[1])
    field_plot(window, vz_field_filtered[3], image_vortices[0], oimage_file,
               'save')

    pvi = [str(timei), str(pvi[0]), str(pvi[1]), str(pvi[2])]
    pvi = ','.join(pvi)
    nvi = [str(timei), str(nvi[0]), str(nvi[1]), str(nvi[2])]
    nvi = ','.join(nvi)
    with open('mpv', 'a') as f:
        f.write("%s\n" % pvi)
    with open('npv', 'a') as f:
        f.write("%s\n" % nvi)
