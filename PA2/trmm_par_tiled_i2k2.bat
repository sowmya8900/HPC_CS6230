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

export OUTFILE=trmm_par_tiled_i2k2.$SLURMD_NODENAME.$SLURM_JOB_ID.log
echo "*** Assigned Granite Node: " $SLURMD_NODENAME | tee -a $OUTFILE
echo " " | tee -a  $OUTFILE
echo "Compiling and executing trmm_par_tiled_i2k2" | tee -a  $OUTFILE
cc -O3 -fopenmp -o trmm_par_tiled_i2k2 mm_main.c mm_ref.c trmm_par_tiled_i2k2.c
./trmm_par_tiled_i2k2 | tee -a  -a $OUTFILE
