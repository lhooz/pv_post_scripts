"""preprocessing scripts for wing kinematics at specified time"""

import os
import csv
import numpy as np
from scipy.interpolate import UnivariateSpline

#----------------------------------------
t_sequence = [4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0]
#----------------------------------------
cwd = os.getcwd()
kinematics_file = os.path.join(cwd, 'backGround/constant/6DoF_motion.dat')
output_file = os.path.join(cwd, 'backGround/paraview/k_sequence.csv')
#----------------------------------------

kinematics_arr = []
with open(kinematics_file) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter='(')
    line_count = 0

    for row in csv_reader:
        if line_count <= 1:
            line_count += 1
        elif row[0] == ')':
            line_count += 1
        else:
            t_datai = row[1]
            # print(row)
            rot_datai0 = row[4].split()[0]
            kinematics_arr.append([
                float(t_datai),
                float(rot_datai0),
            ])
            line_count += 1

    print(f'Processed {line_count} lines in {kinematics_file}')

kinematics_arr = np.array(kinematics_arr)
spl = UnivariateSpline(kinematics_arr[:, 0], kinematics_arr[:, 1], s=0)

with open(output_file, 'w') as f:
    for t in t_sequence:
        phi = spl(t)
        f.write("%s, %s\n" % ('{0:.8f}'.format(t), '{0:.8f}'.format(phi)))
