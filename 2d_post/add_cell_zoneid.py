# trace generated using paraview version 5.7.0
#
#### import the simple module from the paraview
import os
import shutil
# from paraview.simple import *

#### disable automatic camera reset on 'Show'
# paraview.simple._DisableFirstRenderCameraReset()

cwd = os.getcwd()

pvstate_file = os.path.join(cwd, 'backGround/paraview/3d_wing_post.pvsm')
foam_file = os.path.join(cwd, 'backGround/open.foam')
output_folder = os.path.join(cwd, 'backGround/paraview/anime')

if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
# os.mkdir(output_folder)

output_files = os.path.join(output_folder, 'anime.jpeg')

subfolders = [f.name for f in os.scandir(cwd) if f.is_dir()]

processors = []
for folder in subfolders:
    if folder.startswith('processor'):
        processors.append(folder)

for processor in processors:
    ppath = os.path.join(cwd, processor)
    zidpath = os.path.join(ppath, '0/zoneID')

    times = [f.path for f in os.scandir(ppath) if f.is_dir()]
    for time in times:
        if time.endswith('constant') or time.endswith('/0') is False:
            tzidpath = os.path.join(time, 'zoneID')

            shutil.copyfile(zidpath, tzidpath)

"""
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

# save animation
SaveAnimation(
    output_files,
    renderView1,
    ImageResolution=[772, 338],
    OverrideColorPalette='BlackBackground',
    FrameWindow=[0, 799],
    # JPEG options
    Quality=100)
"""
