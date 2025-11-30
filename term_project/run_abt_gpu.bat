#!/bin/bash -x
#SBATCH -M granite
#SBATCH --partition=soc-gpu-class-grn
#SBATCH --qos=soc-gpu-class-grn
#SBATCH --account=cs6230
#SBATCH --gres=gpu:rtxpr6000bl:1
#SBATCH --exclude=grn073
#SBATCH -t 0:5:00
echo "*** Assigned Granite Node: " $SLURMD_NODENAME | tee abt.gpu.$SLURMD_NODENAME.$SLURM_JOB_ID.log
module load cuda
nvcc -O3 -o abt_gpu abt_main.cu abt_launch.cu abt_kernel.cu
./abt_gpu < in1.txt >> abt.gpu.$SLURMD_NODENAME.$SLURM_JOB_ID.log
./abt_gpu < in2.txt >> abt.gpu.$SLURMD_NODENAME.$SLURM_JOB_ID.log
./abt_gpu < in3.txt >> abt.gpu.$SLURMD_NODENAME.$SLURM_JOB_ID.log
./abt_gpu < in4.txt >> abt.gpu.$SLURMD_NODENAME.$SLURM_JOB_ID.log
./abt_gpu < in5.txt >> abt.gpu.$SLURMD_NODENAME.$SLURM_JOB_ID.log
./abt_gpu < in6.txt >> abt.gpu.$SLURMD_NODENAME.$SLURM_JOB_ID.log
./abt_gpu < in7.txt >> abt.gpu.$SLURMD_NODENAME.$SLURM_JOB_ID.log
./abt_gpu < in8.txt >> abt.gpu.$SLURMD_NODENAME.$SLURM_JOB_ID.log
./abt_gpu < in9.txt >> abt.gpu.$SLURMD_NODENAME.$SLURM_JOB_ID.log
./abt_gpu < in10.txt >> abt.gpu.$SLURMD_NODENAME.$SLURM_JOB_ID.log
./abt_gpu < in11.txt >> abt.gpu.$SLURMD_NODENAME.$SLURM_JOB_ID.log
./abt_gpu < in12.txt >> abt.gpu.$SLURMD_NODENAME.$SLURM_JOB_ID.log
./abt_gpu < in13.txt >> abt.gpu.$SLURMD_NODENAME.$SLURM_JOB_ID.log
./abt_gpu < in14.txt >> abt.gpu.$SLURMD_NODENAME.$SLURM_JOB_ID.log
