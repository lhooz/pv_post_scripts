"""plotting cfd run results"""
import os
from functions_plotting import read_cfd_data, read_ref_data, cf_plotter

cwd = os.getcwd()
cfd_data_list = ['sym', 'ad', 'dl']
# ref_data_lst = ['ff_dickinson', 'adff_dickinson', 'dlff_dickinson']
ref_data_lst = []

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
data_to_plot = ['sym', 'dl']
# time_to_plot = 'all'
time_to_plot = [1, 4]

cf_plotter(data_array, legends, data_to_plot, time_to_plot)
