"""postprocessing individual vortices for forceCoefficients and plotting"""

import os
import shutil
import numpy as np

from cprocessing_functions import read_all_vortices, read_origin_ref, plot_v_history, impulse_cf_processing

#--------------------------------------------
#--'all': plot all vortices--
#--N: plot first N long last vortices--
#--list: plot particular vortex in the list--
#--------------------------------------------
# vortices_to_plot = 'all'
# vortices_to_plot = 12
vortices_to_plot = [1, 2, 14, 17]
# vortices_to_plot = [x for x in range(20, 37)]
#--------------------------------------------
time_to_plot = 'all'
# time_to_plot = [0, 1]
#--------------------------------------------
items_to_plot = ['vortices', 'lift']
#------------------------------------------
#--minimum time series length of vortices used for cf calculation--
v_length_lower_limit = 6
#--spline interpolation order for circulations: 1 ~ 5 --
spline_interp_order = 5
#--reference constant for cf calculation--
ref_constant = 5.6816
#------------------------------------------
cwd = os.getcwd()
individual_vortex_folder = os.path.join(cwd, 'individual_vortex_history')
processed_vortices_folder = os.path.join(individual_vortex_folder,
                                         'processed_vortices_history')
origin_ref_folder = os.path.join(individual_vortex_folder, 'origin_ref_folder')
image_out_path = cwd
#------------------------------------------
if os.path.exists(processed_vortices_folder):
    shutil.rmtree(processed_vortices_folder)
os.mkdir(processed_vortices_folder)
origin_ref_file = os.path.join(origin_ref_folder, 'origin_ref')
#------------------------------------------
vor_dict = read_all_vortices(individual_vortex_folder)
origin_ref = read_origin_ref(origin_ref_file)
vor_cf_dict = impulse_cf_processing(vor_dict, ref_constant, origin_ref,
                                    processed_vortices_folder,
                                    v_length_lower_limit, spline_interp_order)
# print(vor_cf_dict)
plot_v_history(vor_cf_dict, image_out_path, vortices_to_plot, time_to_plot,
               items_to_plot)
