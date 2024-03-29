"""preprocessing scripts for wing kinematics at specified time"""

import os
import csv
import numpy as np
from scipy.interpolate import UnivariateSpline

#----------------------------------------
t_sequence = np.linspace(4.05, 5, 20)
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
            rot_datai1 = row[4].split()[1]
            kinematics_arr.append([
                float(t_datai),
                float(rot_datai0),
                float(rot_datai1),
            ])
            line_count += 1

    print(f'Processed {line_count} lines in {kinematics_file}')

kinematics_arr = np.array(kinematics_arr)
spl = UnivariateSpline(kinematics_arr[:, 0], kinematics_arr[:, 1], s=0)
spl1 = UnivariateSpline(kinematics_arr[:, 0], kinematics_arr[:, 2], s=0)

with open(output_file, 'w') as f:
    for t in t_sequence:
        phi = spl(t)
        theta = spl1(t)
        f.write("%s, %s, %s\n" % ('{0:.8f}'.format(t), '{0:.8f}'.format(phi),
                                  '{0:.8f}'.format(theta)))
