#!/bin/bash -x
#SBATCH -M granite
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=coe-class-grn
#SBATCH --qos=coe-class-shortbatch-grn
#SBATCH --account=cs6230
#SBATCH --cpus-per-task=64
#SBATCH -t 0:10:00
echo "*** Assigned Granite Node: " $SLURMD_NODENAME | tee atbt.cpu.$SLURMD_NODENAME.$SLURM_JOB_ID.log
cc -O3 -fopenmp -o atbt_cpu atbt_main.c atbt_par.c
./atbt_cpu < in1.txt >> atbt.cpu.$SLURMD_NODENAME.$SLURM_JOB_ID.log
./atbt_cpu < in2.txt >> atbt.cpu.$SLURMD_NODENAME.$SLURM_JOB_ID.log
./atbt_cpu < in3.txt >> atbt.cpu.$SLURMD_NODENAME.$SLURM_JOB_ID.log
./atbt_cpu < in4.txt >> atbt.cpu.$SLURMD_NODENAME.$SLURM_JOB_ID.log
./atbt_cpu < in5.txt >> atbt.cpu.$SLURMD_NODENAME.$SLURM_JOB_ID.log
./atbt_cpu < in6.txt >> atbt.cpu.$SLURMD_NODENAME.$SLURM_JOB_ID.log
./atbt_cpu < in7.txt >> atbt.cpu.$SLURMD_NODENAME.$SLURM_JOB_ID.log
./atbt_cpu < in8.txt >> atbt.cpu.$SLURMD_NODENAME.$SLURM_JOB_ID.log
./atbt_cpu < in9.txt >> atbt.cpu.$SLURMD_NODENAME.$SLURM_JOB_ID.log
./atbt_cpu < in10.txt >> atbt.cpu.$SLURMD_NODENAME.$SLURM_JOB_ID.log
./atbt_cpu < in11.txt >> atbt.cpu.$SLURMD_NODENAME.$SLURM_JOB_ID.log
./atbt_cpu < in12.txt >> atbt.cpu.$SLURMD_NODENAME.$SLURM_JOB_ID.log
./atbt_cpu < in13.txt >> atbt.cpu.$SLURMD_NODENAME.$SLURM_JOB_ID.log
./atbt_cpu < in14.txt >> atbt.cpu.$SLURMD_NODENAME.$SLURM_JOB_ID.log
