"""plotting cfd run results"""
import os

from functions_plotting import cf_plotter, read_cfd_data, read_ref_data, append_kinematics_array

#-------------input plot control----------
cfd_data_list = [
    'Re100.0_stroke9.0_acf0.25_pf0.125', 'Re100.0_stroke9.0_acf0.25_pf0.25',
    'Re100.0_stroke9.0_acf0.25_pf0.5'
]
# ref_data_lst = ['sym_exp', 'sym_cfd1', 'sym_cfd2']
ref_data_lst = []
#-----------------------------------------
# data_to_plot = [
    # 'Re100.0_stroke9.0_acf0.25_pf0.125', 'Re100.0_stroke9.0_acf0.25_pf0.25',
    # 'Re100.0_stroke9.0_acf0.25_pf0.5'
# ]
data_to_plot = cfd_data_list
# data_to_plot = [cfd_data_list[-1]]
#---------------------------------------
time_to_plot = 'all'
coeffs_show_range = 'all'
time_to_plot = [11.4, 14]
coeffs_show_range = [-1, 3]
cycle_time = 1
#---------------------------------------
cwd = os.getcwd()
image_out_path = cwd
#---------------------------------------
cf_array = []
for cfi in cfd_data_list:
    cfd_datai = os.path.join(cwd, cfi)
    cf_arrayi = read_cfd_data(cfd_datai)

    # cfd_kinematics_datai = os.path.join(cwd, '2d_kinematic_cases/' + cfi + '.dat')
    # cf_arrayi = append_kinematics_array(cf_arrayi, cfd_kinematics_datai)

    cf_array.append(cf_arrayi)

ref_array = []
for refi in ref_data_lst:
    ref_datai = os.path.join(cwd, refi + '.csv')
    ref_arrayi = read_ref_data(ref_datai)

    ref_array.append(ref_arrayi)

data_array = [cf_array, ref_array]
legends = [cfd_data_list, ref_data_lst]
#---------------------------------------

cf_plotter(data_array, legends, data_to_plot, time_to_plot, coeffs_show_range,
           image_out_path, cycle_time, 'against_t')
