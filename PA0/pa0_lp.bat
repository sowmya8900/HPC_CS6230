#!/bin/bash -x
#SBATCH -M lonepeak
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:05:00
#SBATCH --partition=lonepeak-guest 
#SBATCH --qos=lonepeak-guest 
#SBATCH --account=owner-guest 

# Put output into a file whose name starts with that of the allocated CHPC node for the job
export OUTPUT_FILE=$SLURM_SUBMIT_DIR/$SLURMD_NODENAME.$SLURM_JOB_ID.pa0.output
echo "*** Assigned Node: " $SLURMD_NODENAME | tee $OUTPUT_FILE
# Get information about the processor on allocated CHPC node
lscpu | tee -a $OUTPUT_FILE

# Create a unique scratch directory for job
export JOB_TEMP_DIR=/scratch/local/$USER/$SLURM_JOB_ID
mkdir -p $JOB_TEMP_DIR

cd $JOB_TEMP_DIR

# Copy programs to the scratch directory
cp $SLURM_SUBMIT_DIR/*.c .

echo "" | tee -a $OUTPUT_FILE
echo "Compiling and Running e1.c 3 times" | tee -a $OUTPUT_FILE
cc -O3 -o e1 e1.c
./e1 | tee -a $OUTPUT_FILE
./e1 | tee -a $OUTPUT_FILE
./e1 | tee -a $OUTPUT_FILE

echo "" | tee -a $OUTPUT_FILE
echo "Compiling and Running e2.c 3 times" | tee -a $OUTPUT_FILE
cc -O3 -o e2 e2.c
./e2 | tee -a $OUTPUT_FILE
./e2 | tee -a $OUTPUT_FILE
./e2 | tee -a $OUTPUT_FILE

echo "" | tee -a $OUTPUT_FILE
echo "Compiling and Running inc.c 3 times" | tee -a $OUTPUT_FILE
cc -O3 -o inc inc.c
./inc | tee -a $OUTPUT_FILE
./inc | tee -a $OUTPUT_FILE
./inc | tee -a $OUTPUT_FILE

echo "" | tee -a $OUTPUT_FILE
echo "Compiling and Running inc-big.c 3 times" | tee -a $OUTPUT_FILE
cc -O3 -o inc-big inc-big.c
./inc-big | tee -a $OUTPUT_FILE
./inc-big | tee -a $OUTPUT_FILE
./inc-big | tee -a $OUTPUT_FILE
