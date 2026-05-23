#!/bin/bash

core_scripts=(
  "experiments/rl_benchmarking/ppo_nominal.py"
  "experiments/rl_benchmarking/ppo_plain.py"
  "experiments/rl_benchmarking/ppo_adaptive_sim_residual.py"
  "experiments/rl_benchmarking/ppo_adaptive_transformer_residual.py"
  "experiments/rl_benchmarking/ppo_adaptive_transformer_residual_task_predictive.py"
  "experiments/rl_benchmarking/ppo_adaptive_transformer_structured.py"
  "experiments/rl_benchmarking/ppo_adaptive_transformer_task_predictive.py"
  "experiments/rl_benchmarking/ppo_adaptive_cross_attention_structured.py"
  "experiments/rl_benchmarking/ppo_adaptive_cross_attention_task_predictive.py"
)

optional_control_scripts=(
  "experiments/rl_benchmarking/ppo_adaptive_sim_structured.py"
  "experiments/rl_benchmarking/ppo_adaptive_cnn_residual.py"
  "experiments/rl_benchmarking/ppo_adaptive_cnn_structured.py"
  "experiments/rl_benchmarking/ppo_adaptive_cross_attention_residual_task_predictive.py"
)

diagnostic_scripts=(
  "experiments/rl_benchmarking/counterfactual_demand_probe.py"
)

classic_control_scripts=(
  # "experiments/rl_benchmarking/pd_controller.py"
  # "experiments/rl_benchmarking/lqr.py"
  # "experiments/rl_benchmarking/nominal_mpc.py"
)

scripts=("${core_scripts[@]}")
if [ "${RUN_OPTIONAL_CONTROLS:-0}" = "1" ]; then
  scripts+=("${optional_control_scripts[@]}")
fi
if [ "${RUN_CLASSIC_CONTROLS:-0}" = "1" ]; then
  scripts+=("${classic_control_scripts[@]}")
fi
if [ "${RUN_COUNTERFACTUAL_PROBE:-0}" = "1" ]; then
  scripts+=("${diagnostic_scripts[@]}")
fi

# Ensure repository root is on PYTHONPATH so `experiments` can be imported
export PYTHONPATH="$(pwd):${PYTHONPATH}"

# Fast tuning default
export SMALLSAT_ROLLOUT_BACKEND="${SMALLSAT_ROLLOUT_BACKEND:-freeflyer}"
echo "Using RL rollout backend: ${SMALLSAT_ROLLOUT_BACKEND}"

# Deployment runs compile a separate controller/deployment graph and execute many
# stress-test scenarios. Skip them by default for training/tuning sweeps.
if [ "${RUN_DEPLOY:-0}" != "1" ]; then
  export SMALLSAT_SKIP_DEPLOY=1
  echo "Skipping deployment tests. Set RUN_DEPLOY=1 to enable them."
else
  export SMALLSAT_SKIP_DEPLOY=0
fi

# W&B is enabled by default so benchmark runs are recorded.
# Use RUN_WANDB=0 only for local smoke tests where W&B overhead is unwanted.
#   RUN_LOG=1 RUN_VIDEO=1 ./train_test_all.sh
#   RUN_LOG=1 RUN_OPTIONAL_CONTROLS=1 ./train_test_all.sh
#   RUN_DEPLOY=1 RUN_LOG=1 ./train_test_all.sh
#   RUN_COUNTERFACTUAL_PROBE=1 ./train_test_all.sh
#
# Existing checkpoints are reused by the Python runners. Remove the relevant
# checkpoint files if you want to force retraining a variant.
COMMON_ARGS=(--headless)
if [ "${RUN_WANDB:-1}" = "1" ]; then
  COMMON_ARGS+=(--wandb)
fi
if [ "${RUN_LOG:-0}" = "1" ]; then
  COMMON_ARGS+=(--log)
fi
if [ "${RUN_VIDEO:-0}" = "1" ]; then
  COMMON_ARGS+=(--video)
fi

# Iterate over each script
for file in "${scripts[@]}"; do
  if [ -e "$file" ]; then
    echo -e "\nRunning $file...\n"
    python "$file" "${COMMON_ARGS[@]}"
  else
    echo "File $file not found."
    exit 1
  fi
done

echo "All scripts executed successfully."
