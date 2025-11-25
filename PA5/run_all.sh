#!/bin/bash
# run_all.sh - run all PA5 executables with srun

module load cuda

# Common srun options
SRUN="srun -M granite --partition=soc-gpu-class-grn --qos=soc-gpu-class-grn \
      --account=cs6230 --gres=gpu:rtxpr6000bl:1 --exclude=grn073"

# List of executables to run
EXES=(mtmt mtmt-k4 mtmt-j4 mtmt-i4 mtmt-i4j4 mtmt-sm mtmt-sm-i4j4)

for exe in "${EXES[@]}"; do
  if [[ -x ./$exe ]]; then
    echo "=== Running $exe ==="
    $SRUN ./$exe
    echo
  else
    echo "Skipping $exe (not found or not executable)"
  fi
done

