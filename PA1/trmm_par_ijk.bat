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

export OUTFILE=trmm_par_ijk.$SLURMD_NODENAME.$SLURM_JOB_ID.log
echo "*** Assigned Granite Node: " $SLURMD_NODENAME | tee -a $OUTFILE
echo " " | tee -a  $OUTFILE
echo "Compiling and executing trmm_par_i" | tee -a  $OUTFILE
cc -O3 -fopenmp -o trmm_par_i trmm_main.c trmm_ref.c trmm_par_i.c
./trmm_par_i | tee -a  -a $OUTFILE

echo " " | tee -a  $OUTFILE
echo "Compiling and executing trmm_par_j" | tee -a  $OUTFILE
cc -O3 -fopenmp -o trmm_par_j trmm_main.c trmm_ref.c trmm_par_j.c
./trmm_par_j | tee -a  -a $OUTFILE

echo " " | tee -a  $OUTFILE
echo "Compiling and executing trmm_par_k" | tee -a  $OUTFILE
cc -O3 -fopenmp -o trmm_par_k trmm_main.c trmm_ref.c trmm_par_k.c
./trmm_par_k | tee -a  -a $OUTFILE
