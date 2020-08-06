"""circulation processing functions"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath
import scipy.interpolate

cwd = os.getcwd()
vorz_folder = os.path.join(cwd, 'vorz_data')
wgeo_folder = os.path.join(cwd, 'wgeo_data')


def read_vorz(vorz_data_file):
    """read vorticity data"""
    vor_array = []
    with open(vorz_data_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                vor_array.append([float(row[0]), float(row[1]), float(row[3])])
                line_count += 1

        print(f'Processed {line_count} lines in {vorz_data_file}')

    vor_array = np.array(vor_array)
    return vor_array


def read_wgeo(wgeo_data_file):
    """read wing geometry data"""
    wgeo_array = []
    with open(wgeo_data_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                wgeo_array.append([float(row[0]), float(row[1])])
                line_count += 1

        print(f'Processed {line_count} lines in {wgeo_data_file}')

    wgeo_array = np.array(wgeo_array)
    return wgeo_array


def geometry_filter(grid_x, grid_y, wgeo_array):
    """geometry filter for vorticity"""
    filter_wgeo = []
    for x_row, y_row in zip(grid_x, grid_y):
        points = np.array([x, y] for x, y in zip(x_row, y_row))
        path = mpltPath.Path(wgeo_array)
        inside = path.contains_points(points)

        filter_wgeo.append(inside)


def threshold_filter(grid_vz, threshold):
    """threshold filter for vorticity"""
    # print(np.amax(-grid_vz))
    filter_pvorz = (grid_vz >= threshold * np.amax(grid_vz)) * 1
    filter_nvorz = (grid_vz <= -threshold * np.amax(-grid_vz)) * 1
    # print(filter_pvorz)
    grid_pvz = np.multiply(grid_vz, filter_pvorz)
    grid_nvz = np.multiply(grid_vz, filter_nvorz)

    t_grid_vz = [grid_pvz, grid_nvz]

    return t_grid_vz


def grid_vorz(window, resolution, vor_array, tfilter, gfilter):
    """grid interpolation for vorticity data"""
    grid_x, grid_y = np.mgrid[window[0]:window[1]:resolution[0] * 1j,
                              window[2]:window[3]:resolution[1] * 1j]

    grid_vz = scipy.interpolate.griddata(vor_array[:, 0:2],
                                         vor_array[:, 2], (grid_x, grid_y),
                                         method='linear')

    plt.subplot(221)
    plt.imshow(grid_vz.T,
               extent=(window[0], window[1], window[2], window[3]),
               origin='lower')
    plt.title('vorz')
    # plt.gcf().set_size_inches(6, 6)

    plt.subplot(222)
    plt.imshow(grid_pvz.T,
               extent=(window[0], window[1], window[2], window[3]),
               origin='lower')
    plt.title('posiitive_vorz')
    # plt.gcf().set_size_inches(6, 6)

    plt.subplot(223)
    plt.imshow(-grid_nvz.T,
               extent=(window[0], window[1], window[2], window[3]),
               origin='lower')
    plt.title('negative_vorz')
    # plt.gcf().set_size_inches(6, 6)

    plt.show()


# for filei in os.listdir(data_folder):
# os.path.join(data_folder, filei)

vorz_data_file = os.path.join(vorz_folder, 'vorz_0050.csv')
wgeo_data_file = os.path.join(wgeo_folder, 'wgeo_0050.csv')
vor_array = read_vorz(vorz_data_file)
wgeo_array = read_vorz(wgeo_foldero_data_file)

window = [-4, 4, -4, 4]
resolution = [1000, 1000]
threshold = 0.1
grid_vorz(window, resolution, threshold, vor_array)
