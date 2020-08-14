"""main script for circulation processing"""

import os

import numpy as np
from scipy import ndimage

from cprocessing_functions import (field_plot, geometry_filter, grid_vorz,
                                   read_sfield, read_wgeo, threshold_filter,
                                   vorz_processing)

cwd = os.getcwd()
vorz_folder = os.path.join(cwd, 'vorz_data')
q_folder = os.path.join(cwd, 'q_data')
wgeo_folder = os.path.join(cwd, 'wgeo_data')

# for filei in os.listdir(data_folder):
# os.path.join(data_folder, filei)

time_instance = '0072'
vorz_data_file = os.path.join(vorz_folder, 'vorz_' + time_instance + '.csv')
q_data_file = os.path.join(q_folder, 'q_' + time_instance + '.csv')
wgeo_data_file = os.path.join(wgeo_folder, 'wgeo_' + time_instance + '.csv')

vor_array = read_sfield(vorz_data_file)
q_array = read_sfield(q_data_file)
wgeo_array = read_wgeo(wgeo_data_file)

window = [-6, 4, -4, 4]
resolution = [1000, 800]
threshold_q = 0.002
threshold_vorz = 0.02
threshold_circulation = 0.02
wbound_radius = 0
# ----------data filtering (pre processing)----------------------
grid_x, grid_y, grid_vz = grid_vorz(window, resolution, vor_array)
grid_x, grid_y, grid_q = grid_vorz(window, resolution, q_array)

g_filter = geometry_filter(grid_x, grid_y, wgeo_array, wbound_radius)

t_filter_q = threshold_filter(grid_q, threshold_q, 'q')
t_filter_vorz = threshold_filter(grid_vz, threshold_vorz, 'vorticity')
t_filter = np.multiply(t_filter_q, t_filter_vorz)
# t_filter = t_filter_q
# t_filter = t_filter_vorz

vorz_filter = np.multiply(g_filter, t_filter)

grid_vz_filtered = vorz_processing(grid_vz, vorz_filter)

# field_plot(window, grid_vz, grid_vz_filtered)
# field_plot(window, grid_vz, vorz_filter)
#----------------------------------------------------------------

#-----------vortices processing------------------------
s = ndimage.generate_binary_structure(2, 2)
vorz_l = ndimage.label(grid_vz_filtered, structure=s)
# print(vorz_l[1])
#-----------locations----------------
pixel_locations = ndimage.measurements.center_of_mass(
    grid_vz_filtered, vorz_l[0], [x for x in range(1, vorz_l[1] + 1)])
# print(pixel_locations)
dx = (window[1] - window[0]) / resolution[0]
dy = (window[3] - window[2]) / resolution[1]
vorz_locations = []
for loci in pixel_locations:
    v_locix = window[0] + loci[0] * dx
    v_lociy = window[2] + loci[1] * dy
    vorz_locations.append([v_locix, v_lociy])

vorz_locations = np.array(vorz_locations)
#-----------circulations----------------
pixel_sum = ndimage.sum(grid_vz_filtered, vorz_l[0],
                        [x for x in range(1, vorz_l[1] + 1)])
circulations = []
for i in range(len(pixel_sum)):
    circulations.append([pixel_sum[i] * dx * dy, i + 1])
circulations = np.array(circulations)

vortices = np.array(
    [[loc[0], loc[1], vz[0], vz[1]]
     for loc, vz in zip(vorz_locations, circulations)
     if vz[0] >= threshold_circulation * np.amax(circulations[:, 0])
     or vz[0] <= threshold_circulation * -np.amax(-circulations[:, 0])])
pvortices = np.array(
    [[loc[0], loc[1], vz[0], vz[1]]
     for loc, vz in zip(vorz_locations, circulations)
     if vz[0] >= threshold_circulation * np.amax(circulations[:, 0])])
nvortices = np.array(
    [[loc[0], loc[1], vz[0], vz[1]]
     for loc, vz in zip(vorz_locations, circulations)
     if vz[0] <= threshold_circulation * -np.amax(-circulations[:, 0])])
#-------------------------------------------------------------------------
show_vortices = nvortices
print(show_vortices)

circulation_filter = vorz_filter * 0
for vortexi in show_vortices:
    circulation_filter += (vorz_l[0] == vortexi[3]) * 1

field_plot(window, grid_vz, circulation_filter)
