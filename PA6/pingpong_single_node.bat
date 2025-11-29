#!/bin/bash -x
#SBATCH -M granite
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --partition=coe-class-grn
#SBATCH --qos=coe-class-shorttest-grn
#SBATCH --account=cs6230
#SBATCH --cpus-per-task=1
#SBATCH -t 0:5:00
echo "*** Assigned Granite Node: " $SLURMD_NODENAME | tee pingpong.single.$SLURMD_NODENAME.$SLURM_JOB_ID.log
module load openmpi
mpicc -o pingpong pingpong.c 
mpirun -np 2 ./pingpong | tee -a pingpong.single.$SLURMD_NODENAME.$SLURM_JOB_ID.log

