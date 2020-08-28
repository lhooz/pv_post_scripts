"""circulation processing functions"""

import csv
import os
import shutil

import matplotlib.path as mpltPath
import matplotlib.pyplot as plt
import numpy as np
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

    wgeo_bound_x = [-np.amax(-wgeo_array[:, 0]), np.amax(wgeo_array[:, 0])]
    # print(wgeo_bound_x)

    return g_filter, wgeo_bound_x


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

    g_filter, wgeo_bound_x = geometry_filter(grid_x, grid_y, wgeo_array,
                                             wbound_radius)

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

    #-------------------------------------------------------------------------
    #-----organizing outputs------------
    circulation_filter = vorz_filter * 0

    for vortexi in vortices:
        circulation_filter += (vorz_l[0] == vortexi[3]) * 1

    vz_flags = vorz_processing(vorz_l[0], circulation_filter)

    vz_circulations = vortices
    image_vortices = circulation_filter
    vz_field = [grid_vz, vz_flags]
    #-------------------------------------------------------------

    return vz_circulations, image_vortices, vz_field, wgeo_bound_x


def mark_current_vortices(no_of_vortices, timei, v_vanish_dist, vortices_last,
                          vortices_current):
    """
    give number to all current vortices according to numbers in last time step
    """
    marked_vortices_current = []
    for vortex in vortices_current:
        vectori = np.array([[vl[2] - vortex[0], vl[3] - vortex[1]]
                            for vl, vc in zip(vortices_last, vortices_current)
                            ])
        dist_array = np.linalg.norm(vectori, axis=1)
        id_min = dist_array.argmin()
        # print(dist_array[id_min])

        if dist_array[id_min] <= v_vanish_dist:
            vortex_mark = vortices_last[id_min][0]
        else:
            no_of_vortices += 1
            vortex_mark = no_of_vortices

        marked_vortex = [
            vortex_mark, timei, vortex[0], vortex[1], vortex[2], vortex[3]
        ]
        marked_vortices_current.append(marked_vortex)

    marked_vortices_current = np.array(marked_vortices_current)

    return no_of_vortices, marked_vortices_current


def vortices_tracking(no_of_vortices, timei, wgeo_boundx_history,
                      v_vanish_dist_factor, marked_vortices_history,
                      vortices_current):
    """
    tracking algorithm for individual vortex
    """
    no_of_vortices_current = len(vortices_current)

    if len(marked_vortices_history) == 0:
        vortices_0 = []
        for i in range(no_of_vortices_current):
            vortexi = [
                i + 1, 0, vortices_current[i][0], vortices_current[i][1], 0,
                vortices_current[i][3]
            ]
            vortices_0.append(vortexi)
            no_of_vortices += 1

        marked_vortices_history.append(np.array(vortices_0))
        v_vanish_dist = 1
    else:
        ref_wing_movement = np.subtract(wgeo_boundx_history[-1],
                                        wgeo_boundx_history[-2])
        ref_wing_movement = np.array(ref_wing_movement)

        v_vanish_dist = v_vanish_dist_factor * np.amax(
            np.absolute(ref_wing_movement))

    # print(v_vanish_dist)
    vortices_last = marked_vortices_history[-1]
    no_of_vortices, marked_vortices_current = mark_current_vortices(
        no_of_vortices, timei, v_vanish_dist, vortices_last, vortices_current)

    marked_vortices_history.append(marked_vortices_current)
    print(f'No of vortices in current field: {no_of_vortices_current}')
    print(f'Total No of vortices in history: {no_of_vortices} \n')

    return no_of_vortices, marked_vortices_history


def write_individual_vortex(window, time_instance, marked_vortices_history,
                            vz_field, individual_vortex_folder,
                            v_no_save_image):
    """
    write out individual vortex history data files
    """

    vz_field_flags = vz_field[1]
    marked_vortices_current = marked_vortices_history[-1]
    exist_vortices_no = np.unique(marked_vortices_current[:, 0])
    if not v_no_save_image == 0:
        ind_v_image_folder = os.path.join(
            individual_vortex_folder,
            'image_vortex_no_' + str(v_no_save_image))

        if not os.path.exists(ind_v_image_folder):
            os.mkdir(ind_v_image_folder)

        ind_v_image_file = os.path.join(ind_v_image_folder,
                                        'time_' + time_instance + '.png')

    for exist_v_noi in exist_vortices_no:
        ind_vortex_history_file = os.path.join(individual_vortex_folder,
                                               'vortex_no_' + str(exist_v_noi))
        # ---------------merging vortices of the same no --------------
        id_vortexi = np.where(marked_vortices_current[:, 0] == exist_v_noi)[0]

        circulation_v_noi = 0
        weighted_x = 0
        weighted_y = 0
        image_v_noi = vz_field_flags * 0
        for id_for_v_noi in id_vortexi:
            circulation_v_noi += marked_vortices_current[id_for_v_noi][4]
            weighted_x += marked_vortices_current[id_for_v_noi][2]
            weighted_y += marked_vortices_current[id_for_v_noi][3]
            image_v_noi += (vz_field_flags ==
                            marked_vortices_current[id_for_v_noi][-1]) * 1

        merged_x = weighted_x / circulation_v_noi
        merged_y = weighted_y / circulation_v_noi

        ind_vortex_historyi = [
            str(marked_vortices_current[0][1]),
            str(merged_x),
            str(merged_y),
            str(circulation_v_noi)
        ]

        with open(ind_vortex_history_file, 'a') as f:
            f.write("%s\n" % ', '.join(ind_vortex_historyi))

        if v_no_save_image == exist_v_noi:
            field_plot(window, vz_field[0], image_v_noi, ind_v_image_file,
                       'save')
            plt.close()
