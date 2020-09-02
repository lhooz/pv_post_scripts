"""fuctions for plotting cfd run results against ref data"""

import csv
import os
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


def cf_plotter(data_array, legends, data_to_plot, time_to_plot,
               image_out_path):
    """
    function to plot cfd force coefficients results
    """
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
    out_image_file = os.path.join(image_out_path, title + '.png')
    ax.set_title(title)
    ax.legend()

    if time_to_plot != 'all':
        ax.set_xlim(time_to_plot)

    plt.savefig(out_image_file)
    plt.show()

    return fig
