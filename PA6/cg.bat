#!/bin/bash -x
#SBATCH -M granite
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --partition=coe-class-grn
#SBATCH --qos=coe-class-shortbatch-grn
#SBATCH --account=cs6230
#SBATCH --cpus-per-task=1
#SBATCH -t 0:1:00
echo "*** Assigned Granite Node: " $SLURMD_NODENAME | tee cg.$SLURMD_NODENAME.$SLURM_JOB_ID.log
module load openmpi
mpicc -o cg cg_main.c cg_seq.c cg_par.c -lm 
echo "CG on 2 processes" | tee -a cg.$SLURMD_NODENAME.$SLURM_JOB_ID.log
mpirun -np 2 ./cg | tee -a cg.$SLURMD_NODENAME.$SLURM_JOB_ID.log
echo "CG on 4 processes" | tee -a cg.$SLURMD_NODENAME.$SLURM_JOB_ID.log
mpirun -np 4 ./cg | tee -a cg.$SLURMD_NODENAME.$SLURM_JOB_ID.log
echo "CG on 8 processes" | tee -a cg.$SLURMD_NODENAME.$SLURM_JOB_ID.log
mpirun -np 8 ./cg | tee -a cg.$SLURMD_NODENAME.$SLURM_JOB_ID.log
echo "CG on 16 processes" | tee -a cg.$SLURMD_NODENAME.$SLURM_JOB_ID.log
mpirun -np 16 ./cg | tee -a cg.$SLURMD_NODENAME.$SLURM_JOB_ID.log

