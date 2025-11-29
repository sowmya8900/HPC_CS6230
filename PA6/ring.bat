#!/bin/bash -x
#SBATCH -M granite
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --partition=coe-class-grn
#SBATCH --qos=coe-class-shortbatch-grn
#SBATCH --account=cs6230
#SBATCH --cpus-per-task=1
#SBATCH -t 0:1:00
echo "*** Assigned Granite Node: " $SLURMD_NODENAME | tee ring.$SLURMD_NODENAME.$SLURM_JOB_ID.log
module load openmpi
mpicc -o ring ring.c 
mpirun ./ring | tee -a ring.$SLURMD_NODENAME.$SLURM_JOB_ID.log

