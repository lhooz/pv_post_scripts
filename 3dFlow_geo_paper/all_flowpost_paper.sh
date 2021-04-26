#!/bin/bash --login

#$ -pe smp.pe 4
#$ -cwd

#$ -m bea
#$ -M hao.lee0019@yahoo.com

#$ -t 1-160
cd ..
mkdir ./FLOW_IMAGES
readarray -t JOB_DIRS < <(find . -mindepth 1 -maxdepth 1 -name '*Re*' -printf '%P\n')

module load apps/binapps/paraview/5.7.0

TID=$[SGE_TASK_ID-1]
JOBDIR=${JOB_DIRS[$TID]}

cp 3dFlow_geo_paper/3d_flow_structure_paper.pvsm $JOBDIR/backGround/paraview
cp 3dFlow_geo_paper/*.py $JOBDIR

cd $JOBDIR
echo "Running SGE_TASK_ID $SGE_TASK_ID in directory $JOBDIR"
pvbatch pre_kinematics.py
mpiexec -n $NSLOTS pvbatch 3d_pvpost_paper.py | tee log.pvpost_paper
cd ..
mkdir ./FLOW_IMAGES/$JOBDIR
cp -r $JOBDIR/flow_images ./FLOW_IMAGES/$JOBDIR
