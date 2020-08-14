"""circulation processing functions"""

import os
import csv
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath
import scipy.interpolate
from scipy import ndimage


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


def field_plot(window, grid_data, grid_data_processed, oimage_file, mode):
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

    if mode == 'save':
        plt.savefig(oimage_file)
    elif mode == 'show':
        plt.show()


def vortices_processing(files_dir, res_parameters, threshold_parameters):
    """main function for vortices processing"""
    vorz_data_file = files_dir[0]
    q_data_file = files_dir[1]
    wgeo_data_file = files_dir[2]

    window = res_parameters[0]
    resolution = res_parameters[1]

    threshold_q = threshold_parameters[0]
    threshold_vorz = threshold_parameters[1]
    threshold_circulation = threshold_parameters[2]
    wbound_radius = threshold_parameters[3]

    vor_array = read_sfield(vorz_data_file)
    q_array = read_sfield(q_data_file)
    wgeo_array = read_wgeo(wgeo_data_file)
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
    if len(pvortices) > 0:
        pvortices = pvortices[pvortices[:, 2].argsort()[::-1]]

    nvortices = np.array(
        [[loc[0], loc[1], vz[0], vz[1]]
         for loc, vz in zip(vorz_locations, circulations)
         if vz[0] <= threshold_circulation * -np.amax(-circulations[:, 0])])
    if len(nvortices) > 0:
        nvortices = nvortices[nvortices[:, 2].argsort()]
    #-------------------------------------------------------------------------
    #-----organizing outputs------------
    pcirculation_filter = vorz_filter * 0
    ncirculation_filter = vorz_filter * 0

    for vortexi in pvortices:
        pcirculation_filter += (vorz_l[0] == vortexi[3]) * 1
    for vortexi in nvortices:
        ncirculation_filter += (vorz_l[0] == vortexi[3]) * 1
    circulation_filter = pcirculation_filter + ncirculation_filter

    pvz_filtered = vorz_processing(grid_vz, pcirculation_filter)
    nvz_filtered = vorz_processing(grid_vz, ncirculation_filter)
    vz_filtered = pvz_filtered + nvz_filtered

    sorted_vortices = [pvortices, nvortices, vortices]
    image_vortices = [
        pcirculation_filter, ncirculation_filter, circulation_filter
    ]
    vz_field_filtered = [pvz_filtered, nvz_filtered, vz_filtered, grid_vz]
    #-------------------------------------------------------------

    return sorted_vortices, image_vortices, vz_field_filtered
