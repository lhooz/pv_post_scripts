"""plotting cfd run forces coefficient results"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt


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


def cf_plotter(data_array, legends, data_to_plot, cycle_to_plot):
    """
    function to plot cfd force coefficients results
    """
    cf_array = np.array(data_array[0])
    ref_array = np.array(data_array[1])
    cf_legends = np.array(legends[0])
    ref_legends = np.array(legends[1])

    ref_array_shifted = []
    for ref_arrayi in ref_array:
        ref_arrayi = np.array(
            [ref_arrayi[:, 0] + cycle_to_plot - 1, ref_arrayi[:, 1]])
        ref_arrayi = np.transpose(ref_arrayi)

        ref_array_shifted.append(ref_arrayi)
        # print(ref_arrayi)

    cf_plot_id = np.zeros(len(cf_legends))
    ref_plot_id = np.zeros(len(ref_legends))
    for data_to_ploti in data_to_plot:
        cf_plot_id = np.logical_or(cf_plot_id, cf_legends == data_to_ploti)
        ref_plot_id = np.logical_or(ref_plot_id, ref_legends == data_to_ploti)

        # print(ref_legends == data_to_ploti)
    # print(cf_plot_id)

    fig, ax = plt.subplots(1, 1)

    for i in range(len(cf_plot_id)):
        if cf_plot_id[i]:
            ax.plot(cf_array[i][:, 0], cf_array[i][:, 3], label=cf_legends[i])

    for i in range(len(ref_plot_id)):
        if ref_plot_id[i]:
            ax.plot(ref_array_shifted[i][:, 0],
                    ref_array_shifted[i][:, 1],
                    label=ref_legends[i])

    ax.set_xlabel('t (seconds)')
    ax.set_ylabel('lift/cl (N/-)')
    title_sep = ' vs. '
    title = title_sep.join(data_to_plot)
    ax.set_title(title)
    ax.legend()

    ax.set_xlim([cycle_to_plot - 1, cycle_to_plot])

    plt.show()

    return fig


#-------------------------------------------------------------------
cwd = os.getcwd()
cfd_data_list = ['ff', 'adff', 'dlff']
ref_data_lst = ['ff_dickinson', 'adff_dickinson', 'dlff_dickinson']

cf_array = []
for cfi in cfd_data_list:
    cfd_datai = os.path.join(cwd, cfi)
    cf_arrayi = read_cfd_data(cfd_datai)

    cf_array.append(cf_arrayi)

ref_array = []
for refi in ref_data_lst:
    ref_datai = os.path.join(cwd, refi)
    ref_arrayi = read_ref_data(ref_datai)

    ref_array.append(ref_arrayi)

data_array = [cf_array, ref_array]
legends = [cfd_data_list, ref_data_lst]
data_to_plot = ['dlff', 'dlff_dickinson']
cycle_to_plot = 5

cf_plotter(data_array, legends, data_to_plot, cycle_to_plot)
