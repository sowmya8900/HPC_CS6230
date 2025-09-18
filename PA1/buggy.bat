#!/bin/bash -x
#SBATCH -M granite
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=coe-class-grn
#SBATCH --qos=coe-class-shorttest-grn
#SBATCH --account=cs6230
#SBATCH --cpus-per-task=4
#SBATCH -t 0:05:00
echo "*** Assigned Granite Node: " $SLURMD_NODENAME | tee -a buggy.$SLURMD_NODENAME.$SLURM_JOB_ID.log
echo " "
cc -O3 -fopenmp -o buggy buggy.c
./buggy | tee -a buggy.$SLURMD_NODENAME.$SLURM_JOB_ID.log

