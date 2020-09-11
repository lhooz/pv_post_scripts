"""circulation processing functions"""

import csv
import os
import shutil

import matplotlib.path as mpltPath
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
from scipy import ndimage
from scipy.interpolate import InterpolatedUnivariateSpline


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

        t_filter = [filter_pvorz, filter_nvorz]
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
    t_filter_v = np.multiply(t_filter_q, t_filter_vorz[0] + t_filter_vorz[1])
    # t_filter = t_filter_q
    # t_filter = t_filter_vorz

    vorz_filter = np.multiply(g_filter, t_filter_v)

    grid_vz_filtered = vorz_processing(grid_vz, vorz_filter)
    # field_plot(window, grid_vz, grid_vz_filtered)
    #----------------------------------------------------------------

    #-----------vortices processing------------------------
    s = ndimage.generate_binary_structure(2, 2)
    vorz_l = ndimage.label(grid_vz_filtered, structure=s)
    # print(vorz_l[1])
    #-----------locations----------------
    pixel_locations_v = ndimage.measurements.center_of_mass(
        np.absolute(grid_vz_filtered), vorz_l[0],
        [x for x in range(1, vorz_l[1] + 1)])
    # print(pixel_locations_v)
    # print(window[0])
    dx = (window[1] - window[0]) / resolution[0]
    dy = (window[3] - window[2]) / resolution[1]
    vorz_locations = []
    for loci in pixel_locations_v:
        v_locix = window[0] + loci[0] * dx
        v_lociy = window[2] + loci[1] * dy
        vorz_locations.append([v_locix, v_lociy])

    vorz_locations = np.array(vorz_locations)
    # print(pvorz_locations)
    #-----------circulations----------------
    pixel_sum_v = ndimage.sum(grid_vz_filtered, vorz_l[0],
                              [x for x in range(1, vorz_l[1] + 1)])
    circulations_v = []
    for i in range(len(pixel_sum_v)):
        circulations_v.append([pixel_sum_v[i] * dx * dy, i + 1])
    circulations_v = np.array(circulations_v)

    pvortices = np.array(
        [[loc[0], loc[1], vz[0], vz[1]]
         for loc, vz in zip(vorz_locations, circulations_v)
         if vz[0] >= threshold_circulation * np.amax(circulations_v[:, 0])])
    nvortices = np.array(
        [[loc[0], loc[1], vz[0], vz[1]]
         for loc, vz in zip(vorz_locations, circulations_v)
         if vz[0] <= threshold_circulation * -np.amax(-circulations_v[:, 0])])
    # print(pvortices)
    #-------------------------------------------------------------------------
    #-----organizing outputs------------
    circulation_filter = vorz_filter * 0

    for pvortexi in pvortices:
        circulation_filter += (vorz_l[0] == pvortexi[3]) * 1
    for nvortexi in nvortices:
        circulation_filter += (vorz_l[0] == nvortexi[3]) * 1

    vz_flags = vorz_processing(vorz_l[0], circulation_filter)

    vz_circulations = [pvortices, nvortices]
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
                      vortices_current, vortices_type):
    """
    tracking algorithm for individual vortex
    """
    no_of_vortices_current = len(vortices_current)

    if len(marked_vortices_history) == 0:
        vortices_0 = []
        for i in range(no_of_vortices_current):
            no_of_vortices += 1
            vortexi = [
                no_of_vortices, 0, vortices_current[i][0],
                vortices_current[i][1], 0, vortices_current[i][3]
            ]
            vortices_0.append(vortexi)

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
    if vortices_type == 'positive':
        print(
            f'No of positive vortices in current field: {no_of_vortices_current}'
        )
    elif vortices_type == 'negative':

        print(
            f'No of negative vortices in current field: {no_of_vortices_current}'
        )
    elif vortices_type == 'all':

        print(f'No of vortices in current field: {no_of_vortices_current}')

    return no_of_vortices, marked_vortices_history


def write_individual_vortex(window, time_instance, marked_vortices_history,
                            org_vorz_field, vz_field_flags,
                            individual_vortex_folder, v_no_save_image):
    """
    write out individual vortex history data files
    """

    marked_vortices_current = marked_vortices_history[-1]
    exist_vortices_no = np.unique(marked_vortices_current[:, 0])
    # print(exist_vortices_no)
    ind_v_image_file = []
    for v_no_save_imagei in v_no_save_image:
        ind_v_image_folderi = os.path.join(
            individual_vortex_folder,
            'image_vortex_no_' + str(v_no_save_imagei))

        if not os.path.exists(ind_v_image_folderi):
            os.mkdir(ind_v_image_folderi)

        ind_v_image_filei = os.path.join(ind_v_image_folderi,
                                         'time_' + time_instance + '.png')
        ind_v_image_file.append(ind_v_image_filei)

    for exist_v_noi in exist_vortices_no:
        ind_vortex_history_file = os.path.join(
            individual_vortex_folder,
            'vortex_no_' + str(int(exist_v_noi)).zfill(4))
        # ---------------merging vortices of the same no --------------
        id_vortexi = np.where(marked_vortices_current[:, 0] == exist_v_noi)[0]
        # print(id_vortexi)

        circulation_v_noi = 0
        weighted_x_sum = 0
        weighted_y_sum = 0
        image_v_noi = vz_field_flags * 0
        for id_for_v_noi in id_vortexi:
            circulation_v_noi += marked_vortices_current[id_for_v_noi][4]
            # print(circulation_v_noi)
            weighted_x_sum += circulation_v_noi * marked_vortices_current[
                id_for_v_noi][2]
            # print(weighted_x)
            weighted_y_sum += circulation_v_noi * marked_vortices_current[
                id_for_v_noi][3]
            image_v_noi += (vz_field_flags ==
                            marked_vortices_current[id_for_v_noi][-1]) * 1

        merged_x = weighted_x_sum / circulation_v_noi
        merged_y = weighted_y_sum / circulation_v_noi
        #---------------------------------------------------------------

        ind_vortex_historyi = [
            str(marked_vortices_current[0][1]),
            str(merged_x),
            str(merged_y),
            str(circulation_v_noi)
        ]

        with open(ind_vortex_history_file, 'a') as f:
            f.write("%s\n" % ', '.join(ind_vortex_historyi))

        for i in range(len(v_no_save_image)):
            if exist_v_noi == v_no_save_image[i]:
                field_plot(window, org_vorz_field, image_v_noi,
                           ind_v_image_file[i], 'save')
            plt.close()


"""functions for reading and plotting individual vortices"""


def read_all_vortices(individual_vortex_folder):
    """read all vortices in vortex folder"""
    v_names = [
        f.name for f in os.scandir(individual_vortex_folder) if f.is_file()
    ]

    vor_list = []
    vor_array = []
    for vnamei in v_names:
        vpathi = os.path.join(individual_vortex_folder, vnamei)

        vor_array.clear()
        with open(vpathi) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0

            for row in csv_reader:
                vor_array.append([
                    float(row[0]),
                    float(row[1]),
                    float(row[2]),
                    float(row[3])
                ])
                line_count += 1

            print(f'Processed {line_count} lines in {vpathi}')

        vor_array_to_append = np.array(vor_array)
        vor_list.append(vor_array_to_append)
        # print(vor_array)

    vor_dict = dict(zip(v_names, vor_list))
    return vor_dict


def plot_v_history(vor_dict, image_out_path, vortices_to_plot, time_to_plot,
                   items_to_plot):
    """plot circulations and locations of vortices"""

    cwd = os.getcwd()
    plt.rcParams.update({
        # "text.usetex": True,
        'mathtext.fontset': 'stix',
        'font.family': 'STIXGeneral',
        'font.size': 14,
        'figure.figsize': (10, 6),
        'lines.linewidth': 1,
        'lines.markersize': 4,
        'lines.markerfacecolor': 'white',
        'figure.dpi': 100,
    })

    if 'vortices' in items_to_plot:
        fig1, ax1 = plt.subplots(1, 1)
        ax1.set_xlabel('t (s)')
        ax1.set_ylabel('x (m)')
        title1 = 'x_location_history_of_vortices'
        fig1.suptitle(title1)

        fig2, ax2 = plt.subplots(1, 1)
        ax2.set_xlabel('t (s)')
        ax2.set_ylabel('y (m)')
        title2 = 'y_location_history_of_vortices'
        fig2.suptitle(title2)

        fig3, ax3 = plt.subplots(1, 1)
        ax3.set_xlabel('t (s)')
        ax3.set_ylabel(r'$\Gamma\/(m^2/s)$')
        title3 = 'circulation_history_of_vortices'
        fig3.suptitle(title3)

    if 'lift' in items_to_plot:
        fig4, ax4 = plt.subplots(1, 1)
        ax4.set_xlabel('t (s)')
        ax4.set_ylabel('cl')
        title4 = 'lift_coeffs_of_vortices'
        fig4.suptitle(title4)

        fig5, ax5 = plt.subplots(1, 1)
        ax5.set_xlabel('t (s)')
        ax5.set_ylabel('cl')
        title5 = 'total_lift_coeffs'
        fig5.suptitle(title5)

        total_lift_coeffs = np.zeros([1, 2])
        total_lift_coeffs_file = os.path.join(cwd, 'summed_cl')
        with open(total_lift_coeffs_file, 'w') as f:
            f.write("%s\n" % ', '.join([str(x) for x in total_lift_coeffs[0]]))

    if 'drag' in items_to_plot:
        fig6, ax6 = plt.subplots(1, 1)
        ax6.set_xlabel('t (s)')
        ax6.set_ylabel('cd')
        title6 = 'drag_coeffs_of_vortices'
        fig6.suptitle(title6)

        fig7, ax7 = plt.subplots(1, 1)
        ax7.set_xlabel('t (s)')
        ax7.set_ylabel('cd')
        title7 = 'total_drag_coeffs'
        fig7.suptitle(title7)

        total_drag_coeffs = np.zeros([1, 2])
        total_drag_coeffs_file = os.path.join(cwd, 'summed_cd')
        with open(total_drag_coeffs_file, 'w') as f:
            f.write("%s\n" % ', '.join([str(x) for x in total_drag_coeffs[0]]))

    length_arr = []
    for vori in vor_dict.values():
        length_arr.append(-len(vori))
    ind = sorted(range(len(length_arr)), key=lambda k: length_arr[k])
    # print(ind)
    vor_dict = [list(vor_dict.items())[i] for i in ind]
    vor_dict = dict(vor_dict)

    #----------------------------------------------
    if vortices_to_plot == 'all':
        vorid = np.array(list(vor_dict.keys()))
        vortices_to_plot = [int(x.split('_')[-1]) for x in vorid]
        # print(vortices_to_plot)
    elif isinstance(vortices_to_plot, int):
        v_plot_all = np.array(list(vor_dict.keys())[0:vortices_to_plot])
        print('\nfirst ' + str(int(vortices_to_plot)) +
              ' longest last vortices:')
        vortices_to_plot = []
        for item in v_plot_all:
            item_no = int(item.split('_')[-1])
            vortices_to_plot.append(item_no)
            print(item)


#------------------------------------------------
    for vortices_to_ploti in vortices_to_plot:
        v_plot = 'vortex_no_' + str(vortices_to_ploti).zfill(4)
        locx_label = 'v' + str(vortices_to_ploti).zfill(4) + '_x'
        locy_label = 'v' + str(vortices_to_ploti).zfill(4) + '_y'
        circulation_label = 'v' + str(vortices_to_ploti).zfill(
            4) + r'_$\Gamma$'
        cl_label = 'v' + str(vortices_to_ploti).zfill(4) + '_cl'
        cd_label = 'v' + str(vortices_to_ploti).zfill(4) + '_cd'
        cl_total_label = 'cl'
        cd_total_label = 'cd'

        v_list = vor_dict[v_plot]
        v_array = np.array(v_list)
        if 'vortices' in items_to_plot:
            ax1.plot(v_array[:, 0],
                     v_array[:, 1],
                     marker='o',
                     linestyle='dashed',
                     label=locx_label)
            ax2.plot(v_array[:, 0],
                     v_array[:, 2],
                     marker='o',
                     linestyle='dashed',
                     label=locy_label)
            ax3.plot(v_array[:, 0],
                     v_array[:, 3],
                     marker='o',
                     linestyle='dashed',
                     label=circulation_label)

        if 'lift' in items_to_plot:
            ax4.plot(v_array[:, 0],
                     v_array[:, 4],
                     marker='o',
                     linestyle='dashed',
                     label=cl_label)

            for i in range(len(v_array[:, 0])):
                ti = v_array[i, 0]
                if ti in total_lift_coeffs[:, 0]:
                    tind = np.argwhere(total_lift_coeffs[:, 0] == ti)[0]
                    # print(tind)
                    total_lift_coeffs[tind, 1] += v_array[i, 4]
                else:
                    new_time_array = np.array([[v_array[i, 0], v_array[i, 4]]])
                    total_lift_coeffs = np.append(total_lift_coeffs,
                                                  new_time_array,
                                                  axis=0)
            time_arr = total_lift_coeffs[:, 0]
            ind = sorted(range(len(time_arr)), key=lambda k: time_arr[k])
            total_lift_coeffs = np.array([total_lift_coeffs[i] for i in ind])

        if 'drag' in items_to_plot:
            ax6.plot(v_array[:, 0],
                     v_array[:, 5],
                     marker='o',
                     linestyle='dashed',
                     label=cd_label)

            for i in range(len(v_array[:, 0])):
                ti = v_array[i, 0]
                if ti in total_drag_coeffs[:, 0]:
                    tind = np.argwhere(total_drag_coeffs[:, 0] == ti)[0]
                    total_drag_coeffs[tind, 1] += v_array[i, 5]
                else:
                    new_time_array = np.array([[v_array[i, 0], v_array[i, 5]]])
                    total_drag_coeffs = np.append(total_drag_coeffs,
                                                  new_time_array,
                                                  axis=0)
            time_arr = total_drag_coeffs[:, 0]
            ind = sorted(range(len(time_arr)), key=lambda k: time_arr[k])
            total_drag_coeffs = np.array([total_drag_coeffs[i] for i in ind])

    if 'lift' in items_to_plot:
        # print(total_lift_coeffs)
        ax5.plot(total_lift_coeffs[:, 0],
                 total_lift_coeffs[:, 1],
                 marker='o',
                 linestyle='dashed',
                 label=cl_total_label)

        for line in total_lift_coeffs[1:, :]:
            with open(total_lift_coeffs_file, 'a') as f:
                f.write("%s\n" % ', '.join([str(x) for x in line]))

    if 'drag' in items_to_plot:
        ax7.plot(total_drag_coeffs[:, 0],
                 total_drag_coeffs[:, 1],
                 marker='o',
                 linestyle='dashed',
                 label=cd_total_label)

        for line in total_drag_coeffs[1:, :]:
            with open(total_drag_coeffs_file, 'a') as f:
                f.write("%s\n" % ', '.join([str(x) for x in line]))

    if 'vortices' in items_to_plot:
        ax1.legend()
        ax2.legend()
        ax3.legend()
        out_image_file1 = os.path.join(image_out_path, title1 + '.png')
        out_image_file2 = os.path.join(image_out_path, title2 + '.png')
        out_image_file3 = os.path.join(image_out_path, title3 + '.png')
        fig1.savefig(out_image_file1)
        fig2.savefig(out_image_file2)
        fig3.savefig(out_image_file3)

    if 'lift' in items_to_plot:
        ax4.legend()
        ax5.legend()
        out_image_file4 = os.path.join(image_out_path, title4 + '.png')
        out_image_file5 = os.path.join(image_out_path, title5 + '.png')
        fig4.savefig(out_image_file4)
        fig5.savefig(out_image_file5)

    if 'drag' in items_to_plot:
        ax6.legend()
        ax7.legend()
        out_image_file6 = os.path.join(image_out_path, title6 + '.png')
        out_image_file7 = os.path.join(image_out_path, title7 + '.png')
        fig6.savefig(out_image_file6)
        fig7.savefig(out_image_file7)

    if time_to_plot != 'all':
        plt.xlim(time_to_plot)

    plt.show()


def impulse_cf_processing(vor_dict, ref_constant, processed_vortices_folder,
                          v_length_lower_limit):
    """
    function for calculating forces using vortex impulse method
    """
    vorid = np.array(list(vor_dict.keys()))
    vor_history = np.array(list(vor_dict.values()))
    # print(vor_history)

    new_vorid = []
    new_vor_history = []
    for i in range(len(vor_history)):
        voridi = vorid[i]
        vori = vor_history[i]
        processed_vortices_filei = os.path.join(processed_vortices_folder,
                                                voridi)
        if len(vori) >= v_length_lower_limit:
            time = vori[:, 0]
            x_moment = np.multiply(vori[:, 1], vori[:, 3])
            y_moment = np.multiply(vori[:, 2], vori[:, 3])

            xm_spl = InterpolatedUnivariateSpline(time, x_moment)
            ym_spl = InterpolatedUnivariateSpline(time, y_moment)
            xm_res = xm_spl.get_residual()
            ym_res = ym_spl.get_residual()
            print(voridi +
                  f' x_moment spline interpolation residual: {xm_res}')
            print(voridi +
                  f' y_moment spline interpolation residual: {ym_res}')

            clift = []
            cdrag = []
            for ti in time:
                cli = xm_spl.derivatives(ti)[1] / ref_constant
                cdi = ym_spl.derivatives(ti)[1] / ref_constant
                clift.append([cli])
                cdrag.append([cdi])
            clift = np.array(clift)
            cdrag = np.array(cdrag)
            # print(vori)
            # print(clift)

            new_vorid.append(voridi)

            new_vori = np.append(vori, clift, axis=1)
            new_vori = np.append(new_vori, cdrag, axis=1)
            new_vor_history.append(new_vori)

            for new_vori_line in new_vori:
                processed_vortex_historyi = [str(x) for x in new_vori_line]
                with open(processed_vortices_filei, 'a') as f:
                    f.write("%s\n" % ', '.join(processed_vortex_historyi))

    new_vorid = np.array(new_vorid)
    new_vor_history = np.array(new_vor_history)
    vor_cf_dict = dict(zip(new_vorid, new_vor_history))

    return vor_cf_dict
