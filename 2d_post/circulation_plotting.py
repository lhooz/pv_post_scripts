"""plotting individual vortices functions"""

import os
import numpy as np

from cprocessing_functions import read_all_vortices, plot_v_circulations_locs

#--------------------------------------------
#--'all': plot all vortices--
#--N: plot first N long last vortices--
#--list: plot particular vortex in the list--
#--------------------------------------------
# vortices_to_plot = 'all'
# vortices_to_plot = 20
vortices_to_plot = [3, 12, 30]
# vortices_to_plot = [x for x in range(20, 37)]
# time_to_plot = 'all'
time_to_plot = [0, 1]
#------------------------------------------
cwd = os.getcwd()
individual_vortex_folder = os.path.join(cwd, 'individual_vortex_history')
image_out_path = cwd
#------------------------------------------
vor_dict = read_all_vortices(individual_vortex_folder)
# print(vor_dict['vortex_no_0003'])

plot_v_circulations_locs(vor_dict, image_out_path, vortices_to_plot,
                         time_to_plot)
