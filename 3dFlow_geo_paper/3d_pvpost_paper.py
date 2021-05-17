# trace generated using paraview version 5.7.0

import csv
import os
import shutil

import numpy as np
from paraview.simple import *

#-----------------------
Q_contour_value = 2.5
#-----------------------
c_bar = 0.06
#--------------------------------
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()
#--------------------------------
cwd = os.getcwd()
dirname = os.path.basename(cwd)

pvstate_file = os.path.join(cwd, 'backGround', 'paraview',
                            '3d_flow_structure_paper.pvsm')
kinematics_file = os.path.join(cwd, 'backGround', 'paraview', 'k_sequence.csv')
Uref_file = os.path.join(cwd, 'backGround/system/FOforceCoefficients')
foam_file = os.path.join(cwd, 'backGround/open.foam')
flow_image_folder = os.path.join(cwd, 'flow_images')

if os.path.exists(flow_image_folder):
    shutil.rmtree(flow_image_folder)
os.mkdir(flow_image_folder)
#-----------read kinematics--------------
kinematics_arr = []
with open(kinematics_file) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0

    for row in csv_reader:
        t_datai = row[0]
        # print(row)
        rot_datai0 = row[1]
        rot_datai1 = row[2]
        kinematics_arr.append([
            float(t_datai),
            float(rot_datai0),
            float(rot_datai1),
        ])
        line_count += 1

    print(f'Processed {line_count} lines in {kinematics_file}')

kinematics_arr = np.array(kinematics_arr)
#----------read Uref--------------------
with open(Uref_file) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=' ')
    line_count = 0

    for row in csv_reader:
        if line_count == 22:
            Uref = row[5].split(';')[0]
            # print(Uref)
            Uref = float(Uref)
        line_count += 1

print(f'Uref = {Uref}')
Qscale_constant = (c_bar / Uref)**2 / Q_contour_value
#----------------------------------------
# load state
LoadState(pvstate_file,
          LoadStateDataFileOptions='Choose File Names',
          openfoamFileName=foam_file,
          openfoam1FileName=foam_file)

for ki in kinematics_arr:
    time = ki[0]
    phi = ki[1]
    theta = ki[2]
    #--------slice section data----
    print(f'Time = {time}')
    print(f'Flapping angle = {phi}')
    print(f'Pitch angle = {theta}')
    #----------------------------------------
    image_output_files = os.path.join(
        flow_image_folder, 'image_' + '{0:.2f}'.format(time) + '.png')
    #----------------------------------------
    #time
    animationScene1 = GetAnimationScene()
    timeKeeper1 = GetTimeKeeper()
    animationScene1.AnimationTime = time
    timeKeeper1.Time = time
    #---------------------------------------------
    renderView1 = FindViewOrCreate('RenderView1', viewtype='RenderView')
    renderView2 = FindViewOrCreate('RenderView2', viewtype='RenderView')
    renderView3 = FindViewOrCreate('RenderView3', viewtype='RenderView')
    #----------------rot for phi-----------------------------
    rot_field = FindSource('Rot_field')
    SetActiveSource(rot_field)
    rot_field.Transform.Rotate = [-1.0 * phi, 0.0, 0.0]
    rot_wing = FindSource('Rot_wing')
    SetActiveSource(rot_wing)
    rot_wing.Transform.Rotate = [-1.0 * phi, 0.0, 0.0]
    #----------------rot for pitch-----------------------------
    rot_field_P = FindSource('Rot_field_P')
    SetActiveSource(rot_field_P)
    rot_field_P.Transform.Rotate = [0.0, -1.0 * theta, 0.0]
    rot_wing_P = FindSource('Rot_wing_P')
    SetActiveSource(rot_wing_P)
    rot_wing_P.Transform.Rotate = [0.0, -1.0 * theta, 0.0]
    #-------------------------------------------
    q_scaled = FindSource('Q_scaled')
    SetActiveSource(q_scaled)
    q_scaled.Function = 'QField*' + '{0:.8f}'.format(Qscale_constant)
    #-------------------------------------------
    renderView1.Update()
    renderView2.Update()
    renderView3.Update()

    layout1 = GetLayout()

    # save screenshot
    SaveScreenshot(
        image_output_files,
        layout1,
        SaveAllViews=1,
        ImageResolution=[2500, 1356],
        OverrideColorPalette='WhiteBackground',
        # PNG options
        CompressionLevel='4')
