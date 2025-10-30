#!/bin/bash -x
#SBATCH -M granite
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=coe-class-grn
#SBATCH --qos=coe-class-shortbatch-grn
#SBATCH --cpus-per-task=1
#SBATCH --account=cs6230
#SBATCH --exclusive
#SBATCH -t 0:05:00

export OUTFILE=PA3.$SLURMD_NODENAME.$SLURM_JOB_ID.log
echo "*** Assigned Granite Node: " $SLURMD_NODENAME | tee -a $OUTFILE
echo " " | tee -a  $OUTFILE
echo "Compiling and executing vec1" | tee -a  $OUTFILE
clang -O3 -mavx512f -o vec1 vec1_main.c vec1a.c vec1b.c vec1c.c
./vec1 | tee -a  -a $OUTFILE

echo " " | tee -a  $OUTFILE
echo "Compiling and executing vec2" | tee -a  $OUTFILE
clang -O3 -mavx512f -o vec2 vec2_main.c vec2_ref.c vec2_opt.c
./vec2 | tee -a  -a $OUTFILE

echo " " | tee -a  $OUTFILE
echo "Compiling and executing vec3" | tee -a  $OUTFILE
clang -O3 -mavx512f -o vec3 vec3_main.c vec3_ref.c vec3_opt.c
./vec3 | tee -a  -a $OUTFILE

echo " " | tee -a  $OUTFILE
echo "Compiling and executing vec4" | tee -a  $OUTFILE
clang -O3 -mavx512f -o vec4 vec4_main.c vec4_ref.c vec4_opt.c
./vec4 | tee -a  -a $OUTFILE

echo " " | tee -a  $OUTFILE
echo "Compiling and executing vec5" | tee -a  $OUTFILE
clang -O3 -mavx512f -o vec5 vec5_main.c vec5_ref.c vec5_opt.c
./vec5 | tee -a  -a $OUTFILE

