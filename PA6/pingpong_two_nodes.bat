#!/bin/bash -x
#SBATCH -M granite
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --partition=coe-class-grn
#SBATCH --qos=coe-class-shorttest-grn
#SBATCH --account=cs6230
#SBATCH --cpus-per-task=1
#SBATCH -t 0:5:00
CLEAN_NODELIST=$(echo "$SLURM_JOB_NODELIST" | tr -d '[]')
OUTFILE="pingpong_twonodes.${CLEAN_NODELIST}.${SLURM_JOB_ID}.log"
echo "*** Assigned Granite Nodes: " $SLURM_JOB_NODELIST > "$OUTFILE"
module load openmpi
mpicc -o pingpong pingpong.c 
mpirun -np 2 ./pingpong  >> "$OUTFILE"

