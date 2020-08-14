"""circulation processing functions"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath
import scipy.interpolate


def read_sfield(field_data_file):
    """read field (vorticity or q) data"""
    vor_array = []
    with open(field_data_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                vor_array.append([float(row[0]), float(row[1]), float(row[3])])
                line_count += 1

        print(f'Processed {line_count} lines in {field_data_file}')

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
                                         method='cubic')

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


def threshold_filter(grid_t, threshold, datatype):
    """threshold filter for vorticity or q"""
    if datatype == 'vorticity':
        filter_pvorz = (grid_t >= threshold * np.amax(grid_t)) * 1
        filter_nvorz = (grid_t <= -threshold * np.amax(-grid_t)) * 1

        t_filter = filter_pvorz + filter_nvorz
    elif datatype == 'q':
        filter_q = (grid_t >= threshold * np.amax(grid_t)) * 1

        t_filter = filter_q

    return t_filter


def vorz_processing(grid_vz, final_filter):
    """applying filters to vorticity data"""

    grid_vz_final = np.multiply(grid_vz, final_filter)

    return grid_vz_final


def field_plot(window, grid_data, grid_data_processed):
    """plot field data"""
    plt.subplot(121)
    plt.imshow(grid_data.T,
               extent=(window[0], window[1], window[2], window[3]),
               origin='lower')
    plt.title('vorz_original')
    # plt.gcf().set_size_inches(6, 6)

    plt.subplot(122)
    plt.imshow(grid_data_processed.T,
               extent=(window[0], window[1], window[2], window[3]),
               origin='lower')
    plt.title('vorz_processed')
    # plt.gcf().set_size_inches(6, 6)

    plt.show()
