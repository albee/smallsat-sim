#!/bin/bash

# Scripts to run
scripts=("experiments/rl_benchmarking/pd_controller.py" 
        "experiments/rl_benchmarking/nn_controller.py"
        "experiments/rl_benchmarking/nn_controller_adaptive.py"
        "experiments/rl_benchmarking/ppo_no_extrinsics_nominal.py"
        "experiments/rl_benchmarking/ppo_no_extrinsics.py"
        "experiments/rl_benchmarking/ppo_extrinsics_from_sim.py"
        "experiments/rl_benchmarking/ppo_extrinsics_from_am.py"
        "experiments/rl_benchmarking/pretrained_ppo_no_extrinsics_nominal.py"
        # "experiments/rl_benchmarking/pretrained_ppo_no_extrinsics.py" 
        # "experiments/rl_benchmarking/pretrained_ppo_extrinsics_from_sim.py" 
        # "experiments/rl_benchmarking/pretrained_ppo_extrinsics_from_am.py"
        )

# Ensure repository root is on PYTHONPATH so `experiments` can be imported
export PYTHONPATH="$(pwd):${PYTHONPATH}"

# Iterate over each script
for file in "${scripts[@]}"; do
  if [ -e "$file" ]; then
    echo "Running $file..."
    python "$file" --headless --log --wandb
  else
    echo "File $file not found."
    exit 1
  fi
done

echo "All scripts executed successfully."
