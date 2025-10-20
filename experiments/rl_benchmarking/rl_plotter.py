import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from smallsat_sim.utils.logger import Logger


def plot_results(logger: Logger, run_name: str) -> None:
    """
    Plot the results of a simulation run (based on the logged data).
    """
    # Load **all** the logs
    df = logger.load_all_logs(run_id=0)

    if "run_name" not in df.columns:
        print("Column 'run_name' not found in DataFrame")
        return None

    # Get the logs for the current run
    df_run = df[df["run_name"] == run_name]
    
    # Check that the data for this run exists
    if df_run.empty:
        print(f"No data found for run name: {run_name}")
        return

    # Define output directory
    dir_name = "experiments/rl_results/"

    # Define a mapping from stage names to the list of metrics to plot
    stage_metrics_train = {
        "policy_training": [
            "mean_tracking_error",
            "mean_angle_error",
            "mean_extrinsic_error",
            "mean_scaled_episodic_returns",
            "mean_scaled_rewards",
            "actor_loss",
            "critic_loss",
            "num_terminal",
            "mean_log_std",
            "mean_std",
        ],
        "am_training": [
            "mean_tracking_error",
            "mean_angle_error",
            "mean_extrinsic_error",
            "am_train_loss",
            "am_val_loss",
        ],
        "evaluation": [
            "mean_tracking_error",
            "mean_angle_error",
            "mean_extrinsic_error",
            "mean_scaled_episodic_returns",
            "num_terminal",
        ],
    }

    # If the stage column exists, loop over the stage-metrics mapping
    if "stage" in df_run.columns:
        for stage_name, metrics in stage_metrics_train.items():
            stage_data = df_run[df_run["stage"] == stage_name]
            if stage_data.empty:
                print(f"No rows with stage '{stage_name}' found. Nothing to plot.")
            else:
                for metric in metrics:
                    plot_metric(
                        stage_data, metric, dir_name, run_name, stage_name, "Epoch"
                    )

    errors = ["mean_tracking_error", "mean_angle_error", "mean_extrinsic_error"]
    stage_metrics_test = {
        "deployment": errors,
        "stuck_off_deployment": errors,
        "stuck_on_deployment": errors,
        "faulty_valve_deployment": errors,
        "saturated_thrust_deployment": errors,
        "thrust_instability_deployment": errors,
    }

    # If the stage column exists, loop over the stage-metrics mapping
    if "stage" in df_run.columns:
        for stage_name, metrics in stage_metrics_test.items():
            stage_data = df_run[df_run["stage"] == stage_name]
            if stage_data.empty:
                print(f"No rows with stage '{stage_name}' found. Nothing to plot.")
            else:
                for metric in metrics:
                    plot_metric(
                        stage_data, metric, dir_name, run_name, stage_name, "Step"
                    )


def plot_metric(
    df: pd.DataFrame,
    metric_name: str,
    dir_name: str,
    run_name: str,
    stage_name: str,
    xlabel: str,
) -> None:
    """
    Generate and save the plot of a given metric.
    """
    if metric_name not in df.columns:
        return

    # Pull out the non-null values for that metric
    metric = df[metric_name].dropna()

    plt.figure(figsize=(8, 6))
    plt.plot(metric)
    plt.xlabel(xlabel)
    plt.ylabel(metric_name.replace("_", " ").title())
    clean_name = metric_name.replace("_", " ").title()
    plt.title(f"{clean_name} Over All Environments")
    filename = run_name + "_" + stage_name + "_" + metric_name
    plt.savefig(dir_name + filename + ".svg", format="svg")
    plt.savefig(dir_name + filename + ".pdf", format="pdf")


def main():
    """
    Main function to run the plotting script.
    """
    # Create logger
    logger = Logger()

    for run_name in [
        "pd_controller",
        "ppo_no_extrinsics",
        "ppo_extrinsics_from_sim",
        "pretrained_ppo_no_extrinsics",
        "pretrained_ppo_extrinsics_from_sim",
        "ppo_extrinsics_from_am",
        "pretrained_ppo_extrinsics_from_am",
    ]:
        plot_results(logger, run_name)


if __name__ == "__main__":
    main()
