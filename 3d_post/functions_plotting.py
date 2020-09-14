"""fuctions for plotting cfd run results against ref data"""

import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline


def read_cfd_data(cfd_data_file):
    """read cfd results force coefficients data"""
    cf_array = []
    with open(cfd_data_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        line_count = 0

        for row in csv_reader:
            if line_count <= 14:
                line_count += 1
            else:
                cf_array.append([
                    float(row[0]),
                    float(row[1]),
                    float(row[2]),
                    float(row[3]),
                    float(row[4]),
                    float(row[5]),
                    float(row[6])
                ])
                line_count += 1

        print(f'Processed {line_count} lines in {cfd_data_file}')

    cf_array = np.array(cf_array)
    return cf_array


def read_ref_data(ref_data_file):
    """read wing geometry data"""
    ref_array = []
    with open(ref_data_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        for row in csv_reader:
            ref_array.append([float(row[0]), float(row[1])])
            line_count += 1

        print(f'Processed {line_count} lines in {ref_data_file}')

    ref_array = np.array(ref_array)
    return ref_array


def cf_plotter(data_array, legends, data_to_plot, time_to_plot,
               coeffs_show_range, image_out_path, plot_mode):
    """
    function to plot cfd force coefficients results
    """
    plt.rcParams.update({
        # "text.usetex": True,
        'mathtext.fontset': 'stix',
        'font.family': 'STIXGeneral',
        'font.size': 18,
        'figure.figsize': (10, 6),
        'lines.linewidth': 1,
        'lines.markersize': 0.1,
        'lines.markerfacecolor': 'white',
        'figure.dpi': 100,
    })
    cf_array = np.array(data_array[0])
    ref_array = np.array(data_array[1])
    cf_legends = np.array(legends[0])
    ref_legends = np.array(legends[1])

    ref_array_shifted = []
    for ref_arrayi in ref_array:
        if time_to_plot == 'all':
            ref_arrayi = np.array([
                ref_arrayi[:, 0] - np.rint(ref_arrayi[0, 0]), ref_arrayi[:, 1]
            ])
        else:
            ref_arrayi = np.array([
                ref_arrayi[:, 0] - np.rint(ref_arrayi[0, 0]) + time_to_plot[0],
                ref_arrayi[:, 1]
            ])
        ref_arrayi = np.transpose(ref_arrayi)

        ref_array_shifted.append(ref_arrayi)
        # print(ref_arrayi)

    cf_plot_id = np.zeros(len(cf_legends))
    ref_plot_id = np.zeros(len(ref_legends))
    for data_to_ploti in data_to_plot:
        cf_plot_id = np.logical_or(cf_plot_id, cf_legends == data_to_ploti)
        ref_plot_id = np.logical_or(ref_plot_id, ref_legends == data_to_ploti)

        # print(cf_legends == data_to_ploti)
    # print(cf_plot_id)

    fig, ax = plt.subplots(1, 1)
    if plot_mode == 'against_t':
        for i in range(len(cf_plot_id)):
            if cf_plot_id[i]:
                ax.plot(cf_array[i][:, 0],
                        cf_array[i][:, 3],
                        label=cf_legends[i])

        for i in range(len(ref_plot_id)):
            if ref_plot_id[i]:
                ax.plot(ref_array_shifted[i][:, 0],
                        ref_array_shifted[i][:, 1],
                        label=ref_legends[i])

        ax.set_xlabel('t (seconds)')

    elif plot_mode == 'against_phi':
        for i in range(len(cf_plot_id)):
            if cf_plot_id[i]:
                ax.plot(cf_array[i][:, -1],
                        cf_array[i][:, 3],
                        label=cf_legends[i])

        ax.set_xlabel(r'$\phi\/(\deg)$')

    ax.set_ylabel('cl')
    title = 'lift coefficients plot'
    out_image_file = os.path.join(image_out_path, title + '.png')
    ax.set_title(title)
    ax.legend()

    if time_to_plot != 'all':
        ax.set_xlim(time_to_plot)
    if coeffs_show_range != 'all':
        ax.set_ylim(coeffs_show_range)

    plt.savefig(out_image_file)
    plt.show()

    return fig


def append_kinematics_array(cfd_arr, kinematics_data_file):
    """read stroke angle and append to cfd array"""
    kinematics_arr = []
    with open(kinematics_data_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='(')
        line_count = 0

        for row in csv_reader:
            if line_count <= 1:
                line_count += 1
            elif row[0] == ')':
                line_count += 1
            else:
                t_datai = row[1]
                phi_datai = row[-1].split()[0]
                kinematics_arr.append(
                    [float(t_datai), np.abs(float(phi_datai))])
                line_count += 1

        print(f'Processed {line_count} lines in {kinematics_data_file}')

    kinematics_arr = np.array(kinematics_arr)
    phi_spl = UnivariateSpline(kinematics_arr[:, 0], kinematics_arr[:, 1])

    phi = []
    for ti in cfd_arr[:, 0]:
        phii = phi_spl(ti)
        phi.append([phii])
    phi = np.array(phi)

    cfd_arr = np.append(cfd_arr, phi, axis=1)

    return cfd_arr
