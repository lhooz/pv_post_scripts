# trace generated using paraview version 5.7.0
#
#### import the simple module from the paraview
import os
import shutil
from paraview.simple import *

#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

cwd = os.getcwd()

pvstate_file = os.path.join(cwd, '2d_vorticity_post.pvsm')
case_dir = os.path.dirname(cwd)
foam_file = os.path.join(case_dir, 'open.foam')
output_folder = os.path.join(cwd, 'results_data')

if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.mkdir(output_folder)

output_files = os.path.join(output_folder, 'vorz.csv')

# load state
LoadState(pvstate_file,
          LoadStateDataFileOptions='Choose File Names',
          openfoamFileName=foam_file)

# get animation scene
animationScene1 = GetAnimationScene()

# get the time-keeper
timeKeeper1 = GetTimeKeeper()

# find view
renderView1 = FindViewOrCreate('RenderView1', viewtype='RenderView')

# set active view
SetActiveView(renderView1)

# find source
vorZ = FindSource('VorZ')

# set active source
SetActiveSource(vorZ)

# get color transfer function/color map for 'VorZ'
vorZLUT = GetColorTransferFunction('VorZ')

# get opacity transfer function/opacity map for 'VorZ'
vorZPWF = GetOpacityTransferFunction('VorZ')

# save data
SaveData(output_files,
         proxy=vorZ,
         WriteTimeSteps=1,
         Filenamesuffix='_%.4d',
         Precision=10,
         UseScientificNotation=1)
