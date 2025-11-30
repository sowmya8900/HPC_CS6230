#!/bin/bash -x
#SBATCH -M granite
#SBATCH --partition=soc-gpu-class-grn
#SBATCH --qos=soc-gpu-class-grn
#SBATCH --account=cs6230
#SBATCH --gres=gpu:rtxpr6000bl:1
#SBATCH --exclude=grn073
#SBATCH -t 0:5:00
echo "*** Assigned Granite Node: " $SLURMD_NODENAME | tee atbt.gpu.$SLURMD_NODENAME.$SLURM_JOB_ID.log
module load cuda
nvcc -O3 -o atbt_gpu atbt_main.cu atbt_launch.cu atbt_kernel.cu
./atbt_gpu < in1.txt >> atbt.gpu.$SLURMD_NODENAME.$SLURM_JOB_ID.log
./atbt_gpu < in2.txt >> atbt.gpu.$SLURMD_NODENAME.$SLURM_JOB_ID.log
./atbt_gpu < in3.txt >> atbt.gpu.$SLURMD_NODENAME.$SLURM_JOB_ID.log
./atbt_gpu < in4.txt >> atbt.gpu.$SLURMD_NODENAME.$SLURM_JOB_ID.log
./atbt_gpu < in5.txt >> atbt.gpu.$SLURMD_NODENAME.$SLURM_JOB_ID.log
./atbt_gpu < in6.txt >> atbt.gpu.$SLURMD_NODENAME.$SLURM_JOB_ID.log
./atbt_gpu < in7.txt >> atbt.gpu.$SLURMD_NODENAME.$SLURM_JOB_ID.log
./atbt_gpu < in8.txt >> atbt.gpu.$SLURMD_NODENAME.$SLURM_JOB_ID.log
./atbt_gpu < in9.txt >> atbt.gpu.$SLURMD_NODENAME.$SLURM_JOB_ID.log
./atbt_gpu < in10.txt >> atbt.gpu.$SLURMD_NODENAME.$SLURM_JOB_ID.log
./atbt_gpu < in11.txt >> atbt.gpu.$SLURMD_NODENAME.$SLURM_JOB_ID.log
./atbt_gpu < in12.txt >> atbt.gpu.$SLURMD_NODENAME.$SLURM_JOB_ID.log
./atbt_gpu < in13.txt >> atbt.gpu.$SLURMD_NODENAME.$SLURM_JOB_ID.log
./atbt_gpu < in14.txt >> atbt.gpu.$SLURMD_NODENAME.$SLURM_JOB_ID.log
