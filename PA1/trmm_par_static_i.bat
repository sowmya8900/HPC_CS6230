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

export OUTFILE=trmm_static.$SLURMD_NODENAME.$SLURM_JOB_ID.log
echo "*** Assigned Granite Node: " $SLURMD_NODENAME | tee -a $OUTFILE
echo " " | tee -a  $OUTFILE
echo "Compiling and executing trmm_par_i_s_default" | tee -a  $OUTFILE
cc -O3 -fopenmp -o trmm_par_i_s_default trmm_main.c trmm_ref.c trmm_par_i_s_default.c
./trmm_par_i_s_default | tee -a  -a $OUTFILE

echo " " | tee -a  $OUTFILE
echo "Compiling and executing trmm_par_i_s4" | tee -a  $OUTFILE
cc -O3 -fopenmp -o trmm_par_i_s4 trmm_main.c trmm_ref.c trmm_par_i_s4.c
./trmm_par_i_s4 | tee -a  -a $OUTFILE

echo " " | tee -a  $OUTFILE
echo "Compiling and executing trmm_par_i_s1" | tee -a  $OUTFILE
cc -O3 -fopenmp -o trmm_par_i_s1 trmm_main.c trmm_ref.c trmm_par_i_s1.c
./trmm_par_i_s1 | tee -a  -a $OUTFILE
