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

    # ----- sorting points in clockwise order ---
    x = wgeo_array[:, 0]
    y = wgeo_array[:, 1]
    cx = np.mean(x)
    cy = np.mean(y)
    a = np.arctan2(y - cy, x - cx)
    order = a.ravel().argsort()
    x = x[order]
    y = y[order]
    wgeo_array = np.vstack((x, y))

    wgeo_array = np.transpose(wgeo_array)

    return wgeo_array


def grid_vorz(window, resolution, vor_array):
    """grid interpolation for vorticity data"""
    grid_x, grid_y = np.mgrid[window[0]:window[1]:resolution[0] * 1j,
                              window[2]:window[3]:resolution[1] * 1j]

    grid_vz = scipy.interpolate.griddata(vor_array[:, 0:2],
                                         vor_array[:, 2], (grid_x, grid_y),
                                         method='linear')

    return grid_x, grid_y, grid_vz


def geometry_filter(grid_x, grid_y, wgeo_array, wbound_radius):
    """geometry filter for vorticity"""
    filter_wgeo = []
    for x_row, y_row in zip(grid_x, grid_y):
        points = np.array([[x, y] for x, y in zip(x_row, y_row)])
        # print(wgeo_array)
        path = mpltPath.Path(wgeo_array, closed=True)
        outside = np.logical_not(
            path.contains_points(points, radius=wbound_radius))

        filter_wgeo.append(outside)

    g_filter = np.array(filter_wgeo) * 1
    # print(g_filter)
    # with open('test_filter.csv', 'w', newline='') as file:
    # writer = csv.writer(file)
    # writer.writerows(g_filter)

    return g_filter


def threshold_filter(grid_vz, threshold):
    """threshold filter for vorticity"""
    # print(np.amax(-grid_vz))
    filter_pvorz = (grid_vz >= threshold * np.amax(grid_vz)) * 1
    filter_nvorz = (grid_vz <= -threshold * np.amax(-grid_vz)) * 1
    # print(filter_pvorz)

    t_filter = [filter_pvorz, filter_nvorz]

    return t_filter


def vorz_processing(grid_vz, final_filter):
    """applying filters to vorticity data"""

    grid_vz_final = np.multiply(grid_vz, final_filter)

    plt.subplot(121)
    plt.imshow(grid_vz.T,
               extent=(window[0], window[1], window[2], window[3]),
               origin='lower')
    plt.title('vorz')
    # plt.gcf().set_size_inches(6, 6)

    plt.subplot(122)
    plt.imshow(grid_vz_final.T,
               extent=(window[0], window[1], window[2], window[3]),
               origin='lower')
    plt.title('filtered_vorz')
    # plt.gcf().set_size_inches(6, 6)

    plt.show()


# for filei in os.listdir(data_folder):
# os.path.join(data_folder, filei)

vorz_data_file = os.path.join(vorz_folder, 'vorz_0015.csv')
wgeo_data_file = os.path.join(wgeo_folder, 'wgeo_0015.csv')
vor_array = read_vorz(vorz_data_file)
wgeo_array = read_wgeo(wgeo_data_file)

window = [-4, 4, -4, 4]
resolution = [1000, 1000]
threshold = 0.05
wbound_radius = 0.1

grid_x, grid_y, grid_vz = grid_vorz(window, resolution, vor_array)
g_filter = geometry_filter(grid_x, grid_y, wgeo_array, wbound_radius)
t_filter = threshold_filter(grid_vz, threshold)

pvorz_filter = np.multiply(g_filter, t_filter[0])
nvorz_filter = np.multiply(g_filter, t_filter[1])

vorz_processing(grid_vz, pvorz_filter)
