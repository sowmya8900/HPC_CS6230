#!/bin/bash -x
#SBATCH -M granite
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=coe-class-grn
#SBATCH --qos=coe-class-shortbatch-grn
#SBATCH --account=cs6230
#SBATCH --cpus-per-task=64
#SBATCH -t 0:10:00
echo "*** Assigned Granite Node: " $SLURMD_NODENAME | tee abt.cpu.$SLURMD_NODENAME.$SLURM_JOB_ID.log
cc -O3 -fopenmp -o abt_cpu abt_main.c abt_par.c
./abt_cpu < in1.txt >> abt.cpu.$SLURMD_NODENAME.$SLURM_JOB_ID.log
./abt_cpu < in2.txt >> abt.cpu.$SLURMD_NODENAME.$SLURM_JOB_ID.log
./abt_cpu < in3.txt >> abt.cpu.$SLURMD_NODENAME.$SLURM_JOB_ID.log
./abt_cpu < in4.txt >> abt.cpu.$SLURMD_NODENAME.$SLURM_JOB_ID.log
./abt_cpu < in5.txt >> abt.cpu.$SLURMD_NODENAME.$SLURM_JOB_ID.log
./abt_cpu < in6.txt >> abt.cpu.$SLURMD_NODENAME.$SLURM_JOB_ID.log
./abt_cpu < in7.txt >> abt.cpu.$SLURMD_NODENAME.$SLURM_JOB_ID.log
./abt_cpu < in8.txt >> abt.cpu.$SLURMD_NODENAME.$SLURM_JOB_ID.log
./abt_cpu < in9.txt >> abt.cpu.$SLURMD_NODENAME.$SLURM_JOB_ID.log
./abt_cpu < in10.txt >> abt.cpu.$SLURMD_NODENAME.$SLURM_JOB_ID.log
./abt_cpu < in11.txt >> abt.cpu.$SLURMD_NODENAME.$SLURM_JOB_ID.log
./abt_cpu < in12.txt >> abt.cpu.$SLURMD_NODENAME.$SLURM_JOB_ID.log
./abt_cpu < in13.txt >> abt.cpu.$SLURMD_NODENAME.$SLURM_JOB_ID.log
./abt_cpu < in14.txt >> abt.cpu.$SLURMD_NODENAME.$SLURM_JOB_ID.log
