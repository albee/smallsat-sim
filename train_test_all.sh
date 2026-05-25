#!/bin/bash

core_scripts=(
  "experiments/rl_benchmarking/ppo_nominal.py"
  "experiments/rl_benchmarking/ppo_plain.py"
  "experiments/rl_benchmarking/ppo_adaptive_sim_residual.py"
  "experiments/rl_benchmarking/ppo_adaptive_transformer_residual.py"
  "experiments/rl_benchmarking/ppo_adaptive_cross_attention_task_predictive.py"
)

optional_control_scripts=(
  "experiments/rl_benchmarking/ppo_adaptive_sim_structured.py"
  "experiments/rl_benchmarking/ppo_adaptive_cnn_residual.py"
  "experiments/rl_benchmarking/ppo_adaptive_cnn_structured.py"
  "experiments/rl_benchmarking/ppo_adaptive_transformer_residual_task_predictive.py"
  "experiments/rl_benchmarking/ppo_adaptive_transformer_structured.py"
  "experiments/rl_benchmarking/ppo_adaptive_transformer_task_predictive.py"
  "experiments/rl_benchmarking/ppo_adaptive_cross_attention_structured.py"
  "experiments/rl_benchmarking/ppo_adaptive_cross_attention_residual_task_predictive.py"
)

diagnostic_scripts=(
  "experiments/rl_benchmarking/counterfactual_demand_probe.py"
)

scenario_split_eval_scripts=(
  "experiments/rl_benchmarking/evaluate_scenario_splits.py"
)

classic_control_scripts=(
  # "experiments/rl_benchmarking/pd_controller.py"
  # "experiments/rl_benchmarking/lqr.py"
  # "experiments/rl_benchmarking/nominal_mpc.py"
)

if [ "${RUN_ONLY_SCENARIO_SPLIT_EVAL:-0}" = "1" ]; then
  scripts=("${scenario_split_eval_scripts[@]}")
else
  scripts=("${core_scripts[@]}")
fi
if [ "${RUN_OPTIONAL_CONTROLS:-0}" = "1" ]; then
  scripts+=("${optional_control_scripts[@]}")
fi
if [ "${RUN_CLASSIC_CONTROLS:-0}" = "1" ]; then
  scripts+=("${classic_control_scripts[@]}")
fi
if [ "${RUN_COUNTERFACTUAL_PROBE:-0}" = "1" ]; then
  scripts+=("${diagnostic_scripts[@]}")
fi
if [ "${RUN_SCENARIO_SPLIT_EVAL:-0}" = "1" ] && [ "${RUN_ONLY_SCENARIO_SPLIT_EVAL:-0}" != "1" ]; then
  scripts+=("${scenario_split_eval_scripts[@]}")
fi

# Ensure repository root is on PYTHONPATH so `experiments` can be imported
export PYTHONPATH="$(pwd):${PYTHONPATH}"

# Fast tuning default
export SMALLSAT_ROLLOUT_BACKEND="${SMALLSAT_ROLLOUT_BACKEND:-freeflyer}"
echo "Using RL rollout backend: ${SMALLSAT_ROLLOUT_BACKEND}"

# The focused paper suite trains 600 nominal epochs, then advances through
# control-theoretic authority regimes: redundant, marginal, authority-limited,
# and bias-limited/disturbance. The older difficulty and failure-type curricula
# remain available by setting failure_curriculum_mode to "difficulty" or "type"
# in src/smallsat_sim/envs/astrobee_rl/cfg/config.py.

# Deployment runs compile a separate controller/deployment graph and execute many
# stress-test scenarios. Skip them by default for training/tuning sweeps.
if [ "${RUN_DEPLOY:-0}" != "1" ]; then
  export SMALLSAT_SKIP_DEPLOY=1
  echo "Skipping deployment tests. Set RUN_DEPLOY=1 to enable them."
else
  export SMALLSAT_SKIP_DEPLOY=0
fi

# Policy evaluation runs full rollout episodes after training. Skip by default
# for fast training sweeps; enable it when collecting comparable eval numbers.
if [ "${RUN_EVAL:-0}" != "1" ]; then
  export SMALLSAT_SKIP_EVAL=1
  echo "Skipping policy evals. Set RUN_EVAL=1 to enable them."
else
  export SMALLSAT_SKIP_EVAL=0
fi

# W&B is enabled by default so benchmark runs are recorded.
# Use RUN_WANDB=0 only for local smoke tests where W&B overhead is unwanted.
#   RUN_LOG=1 RUN_VIDEO=1 ./train_test_all.sh
#   RUN_LOG=1 RUN_OPTIONAL_CONTROLS=1 ./train_test_all.sh
#   RUN_EVAL=1 RUN_LOG=1 ./train_test_all.sh
#   RUN_DEPLOY=1 RUN_LOG=1 ./train_test_all.sh
#   RUN_COUNTERFACTUAL_PROBE=1 ./train_test_all.sh
#   RUN_SCENARIO_SPLIT_EVAL=1 ./train_test_all.sh
#   RUN_ONLY_SCENARIO_SPLIT_EVAL=1 ./train_test_all.sh
#
# The scenario evaluator defaults to ppo_plain. To evaluate an adaptive
# checkpoint on the same generic and targeted authority-regime/eval/stress
# splits, run it directly, e.g.:
#   python experiments/rl_benchmarking/evaluate_scenario_splits.py --headless --wandb \
#     --run-name ppo_adaptive_sim_residual \
#     --use-adaptive-approach --adaptive-context-mode residual --phase 1
#   python experiments/rl_benchmarking/evaluate_scenario_splits.py --headless --wandb \
#     --run-name ppo_adaptive_cross_attention_task_predictive \
#     --use-adaptive-approach --am-architecture transformer_cross_attention \
#     --adaptive-context-mode structured --task-conditioned --phase 2 \
#     --predict-delta-weight 0.1 --predict-tracking-weight 0.1 \
#     --predict-authority-weight 0.1
# Targeted rows start each scenario from the position/attitude error that
# requires its weakest force/torque authority with failures active from reset;
# use --skip-targeted only for quick smoke tests.
#
# Existing full-pose checkpoints are reused by the Python runners. Adaptive
# variants warm-start from training_state_full_pose_nominal.pkl when it exists,
# with extra context-input rows initialized to zero for fair comparison. Remove
# the relevant checkpoint files if you want to force retraining a variant.
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
