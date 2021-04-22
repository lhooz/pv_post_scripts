# trace generated using paraview version 5.7.0

import csv
import os
import shutil

import numpy as np
from paraview.simple import *

#-----section locations--------
sec_loc = [0.2, 0.4, 0.6, 0.8]
#------------------------------
chord = 0.06
cwd = os.getcwd()
dirname = os.path.basename(cwd)
ar = float(dirname.split('_')[0].split('r')[-1])
ofs = float(dirname.split('_')[1].split('s')[-1])
wspan = ar * chord
ofs_dist = ofs * wspan
sec_dist = []
for sec in sec_loc:
    sec_disti = sec * wspan + ofs_dist
    sec_dist.append(sec_disti)
#---------------------------------------
pvstate_file = os.path.join(cwd, 'backGround', 'paraview', '3d_wing_section_post.pvsm')
kinematics_file = os.path.join(cwd, 'backGround', 'paraview', 'k_sequence.csv')
foam_file = os.path.join(cwd, 'backGround/open.foam')
result_data_folder = os.path.join(cwd, 'processed_sec_data')

if os.path.exists(result_data_folder):
    shutil.rmtree(result_data_folder)
os.mkdir(result_data_folder)
#-----------read kinematics--------------
kinematics_arr = []
with open(kinematics_file) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0

    for row in csv_reader:
        t_datai = row[0]
        # print(row)
        rot_datai0 = row[1]
        kinematics_arr.append([
            float(t_datai),
            float(rot_datai0),
        ])
        line_count += 1

    print(f'Processed {line_count} lines in {kinematics_file}')

kinematics_arr = np.array(kinematics_arr)
#----------------------------------------
# load state
LoadState(pvstate_file,
          LoadStateDataFileOptions='Choose File Names',
          openfoamFileName=foam_file,
          openfoam1FileName=foam_file)
#----------------------------------------

for sec_loci, sec_disti in zip(sec_loc, sec_dist):
    sec_folder = os.path.join(result_data_folder,
                              'section_' + '{0:.2f}'.format(sec_loci))
    field_output_folder = os.path.join(sec_folder, 'field_data')
    geop_output_folder = os.path.join(sec_folder, 'geop_data')

    if os.path.exists(sec_folder):
        shutil.rmtree(sec_folder)
    os.mkdir(sec_folder)
    if os.path.exists(field_output_folder):
        shutil.rmtree(field_output_folder)
    os.mkdir(field_output_folder)
    if os.path.exists(geop_output_folder):
        shutil.rmtree(geop_output_folder)
    os.mkdir(geop_output_folder)

    print(f'\nSection loc. = {sec_loci}')
    print(f'Section dist. = {sec_disti}\n')

    for ki in kinematics_arr:
        time = ki[0]
        phi = ki[1]
        #--------slice section data----
        print(f'Time = {time}')
        print(f'Flapping angle = {phi}')
        #------------------------------

        field_output_files = os.path.join(
            field_output_folder, 'field_' + '{0:.2f}'.format(time) + '.csv')
        geop_output_files = os.path.join(
            geop_output_folder, 'geop_' + '{0:.2f}'.format(time) + '.csv')

        # find view
        spreadSheetView1 = FindViewOrCreate('SpreadSheetView1',
                                            viewtype='SpreadSheetView')
        SetActiveView(spreadSheetView1)
        #time
        animationScene1 = GetAnimationScene()
        timeKeeper1 = GetTimeKeeper()
        animationScene1.AnimationTime = time
        timeKeeper1.Time = time
        #-----------field data processing------
        clip1 = FindSource('Clip1')
        SetActiveSource(clip1)
        #--------------------------------------
        transform1 = Transform(Input=clip1)
        transform1.Transform = 'Transform'
        transform1.Transform.Rotate = [-1 * phi, 0.0, 0.0]
        # create a new 'Slice'
        slice1 = Slice(Input=transform1)
        slice1.SliceType = 'Plane'
        slice1.SliceOffsetValues = [0.0]
        slice1.SliceType.Origin = [0.0, sec_disti, 0.0]
        slice1.SliceType.Normal = [0.0, 1.0, 0.0]
        #--------------------
        SetActiveSource(slice1)
        slice1Display = Show(slice1, spreadSheetView1)
        spreadSheetView1.Update()
        # save data
        SaveData(field_output_files,
                 proxy=slice1,
                 Precision=10,
                 UseScientificNotation=1)
        #--------------------
        Delete(slice1)
        del slice1
        Delete(transform1)
        del transform1

        #----------wing geometry data processing-----
        # find source
        wing = FindSource('wing')
        SetActiveSource(wing)
        #----------------------------------
        transform2 = Transform(Input=wing)
        transform2.Transform = 'Transform'
        transform2.Transform.Rotate = [-1 * phi, 0.0, 0.0]
        # create a new 'Slice'
        slice2 = Slice(Input=transform2)
        slice2.SliceType = 'Plane'
        slice2.SliceOffsetValues = [0.0]
        slice2.SliceType.Origin = [0.0, sec_disti, 0.0]
        slice2.SliceType.Normal = [0.0, 1.0, 0.0]
        # show data in view
        SetActiveSource(slice2)
        slice2Display = Show(slice2, spreadSheetView1)
        spreadSheetView1.Update()
        # save data
        SaveData(geop_output_files,
                 proxy=slice2,
                 Precision=10,
                 UseScientificNotation=1)
        #--------------------
        Delete(slice2)
        del slice2
        Delete(transform2)
        del transform2
