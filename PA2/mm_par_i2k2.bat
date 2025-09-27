#!/bin/bash -x
#SBATCH -M granite
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=coe-class-grn
#SBATCH --qos=coe-class-shortbatch-grn
#SBATCH --cpus-per-task=64
#SBATCH --account=cs6230
#SBATCH --exclusive
#SBATCH -t 0:05:00

export OUTFILE=mm_per.$SLURMD_NODENAME.$SLURM_JOB_ID.log
echo "*** Assigned Granite Node: " $SLURMD_NODENAME | tee -a $OUTFILE
echo " " | tee -a  $OUTFILE
echo "Compiling and executing mm_par_i2k2" | tee -a  $OUTFILE
cc -O3 -fopenmp -o mm_par_i2k2 mm_main.c mm_ref.c mm_par_i2k2.c
./mm_par_i2k2 | tee -a  -a $OUTFILE
