from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from smallsat_sim.utils.logger import Logger


RESULTS_DIR = Path("experiments/rl_results/")

ERROR_METRICS = [
    "mean_tracking_error",
    "mean_angle_error",
    "mean_extrinsic_error",
]

STAGE_METRICS_TRAIN = {
    "policy_training": [
        "mean_tracking_error",
        "mean_angle_error",
        "mean_extrinsic_error",
        "mean_episodic_returns",
        "mean_rewards",
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
        "mean_episodic_returns",
        "num_terminal",
    ],
}

DEPLOYMENT_STAGES = [
    "deployment",
    "stuck_off_deployment",
    "stuck_on_deployment",
    "faulty_valve_deployment",
    "saturated_thrust_deployment",
    "thrust_instability_deployment",
    "constant_force_disturbances_deployment",
]

STAGE_METRICS_TEST = {stage: ERROR_METRICS for stage in DEPLOYMENT_STAGES}


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

    # Define base output directory and keep runs grouped by stage
    dir_name = RESULTS_DIR

    # If the stage column exists, loop over the stage-metrics mapping
    if "stage" in df_run.columns:
        for stage_name, metrics in STAGE_METRICS_TRAIN.items():
            stage_data = df_run[df_run["stage"] == stage_name]
            if stage_data.empty:
                print(f"No rows with stage '{stage_name}' found. Nothing to plot.")
            else:
                for metric in metrics:
                    if stage_name == "evaluation":
                        xlabel = "Eval"
                    elif metric == "mean_episodic_returns":
                        xlabel = "Episode"
                    else:
                        xlabel = "Epoch"

                    plot_metric(
                        stage_data,
                        metric,
                        dir_name,
                        run_name,
                        stage_name,
                        xlabel,
                        x_col="step",
                    )

    # If the stage column exists, loop over the stage-metrics mapping
    if "stage" in df_run.columns:
        for stage_name, metrics in STAGE_METRICS_TEST.items():
            stage_data = df_run[df_run["stage"] == stage_name]
            if stage_data.empty:
                print(f"No rows with stage '{stage_name}' found. Nothing to plot.")
            else:
                for metric in metrics:
                    plot_metric(
                        stage_data,
                        metric,
                        dir_name,
                        run_name,
                        stage_name,
                        "Time (s)",
                        x_col="timestamp",
                        to_seconds=True,
                    )


def plot_metric(
    df: pd.DataFrame,
    metric_name: str,
    base_dir: Path,
    run_name: str,
    stage_name: str,
    xlabel: str,
    x_col: str | None = None,
    to_seconds: bool = False,
) -> None:
    """
    Generate and save the plot of a given metric.
    """
    if metric_name not in df.columns:
        return

    resolved_x_col = _resolve_column_name(df, x_col)
    if x_col and resolved_x_col is None:
        print(
            f"Column '{x_col}' not found in DataFrame. Skipping plot for {metric_name}."
        )
        return

    columns = [metric_name]
    if resolved_x_col:
        columns.append(resolved_x_col)

    metric_df = df[columns].dropna()
    if metric_df.empty:
        return

    metric_series = metric_df[metric_name]

    if resolved_x_col:
        x_series = _prepare_x_data(metric_df[resolved_x_col], to_seconds=to_seconds)
        valid_mask = x_series.notna()
        metric_series = metric_series[valid_mask]
        x_series = x_series[valid_mask]
        if metric_series.empty:
            return
    else:
        x_series = metric_series.index
        if metric_series.empty:
            return

    output_dir = base_dir / run_name / stage_name
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.plot(x_series, metric_series)
    plt.xlabel(xlabel)
    plt.ylabel(metric_name.replace("_", " ").title())
    clean_name = metric_name.replace("_", " ").title()
    plt.title(f"{clean_name} Over All Environments")
    filename = metric_name
    plt.savefig(output_dir / f"{filename}.svg", format="svg")
    plt.savefig(output_dir / f"{filename}.pdf", format="pdf")
    plt.close()


def plot_metric_multi_run(
    df: pd.DataFrame,
    metric_name: str,
    base_dir: Path,
    stage_name: str,
    xlabel: str,
    x_col: str | None,
    to_seconds: bool = False,
) -> None:
    """
    Plot the specified metric for all run names present in the DataFrame.
    """
    if metric_name not in df.columns:
        return
    if "run_name" not in df.columns:
        print("Column 'run_name' not found in DataFrame. Skipping multi-run plot.")
        return

    resolved_x_col = _resolve_column_name(df, x_col)
    if x_col and resolved_x_col is None:
        print(
            f"Column '{x_col}' not found in DataFrame. Skipping multi-run plot for {metric_name}."
        )
        return

    plt.figure(figsize=(8, 6))
    plotted = False

    for run_name, run_df in df.groupby("run_name"):
        columns = [metric_name]
        if resolved_x_col:
            columns.append(resolved_x_col)

        run_metric_df = run_df[columns].dropna()
        if run_metric_df.empty:
            continue

        if resolved_x_col:
            run_metric_df = run_metric_df.sort_values(resolved_x_col)
            metric_series = run_metric_df[metric_name]
            x_series = _prepare_x_data(
                run_metric_df[resolved_x_col], to_seconds=to_seconds
            )
            valid_mask = x_series.notna()
            metric_series = metric_series[valid_mask]
            x_series = x_series[valid_mask]
        else:
            metric_series = run_metric_df[metric_name]
            x_series = metric_series.index

        if metric_series.empty:
            continue

        plt.plot(x_series, metric_series, label=run_name)
        plotted = True

    if not plotted:
        plt.close()
        return

    plt.xlabel(xlabel)
    plt.ylabel(metric_name.replace("_", " ").title())
    clean_name = metric_name.replace("_", " ").title()
    stage_label = stage_name.replace("_", " ").title()
    plt.title(f"{clean_name} Across Runs ({stage_label})")
    plt.legend()

    output_dir = base_dir / "all_runs" / stage_name
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{metric_name}_all_runs"
    plt.savefig(output_dir / f"{filename}.svg", format="svg")
    plt.savefig(output_dir / f"{filename}.pdf", format="pdf")
    plt.close()


def plot_deployment_errors_all_runs(logger: Logger) -> None:
    """
    Generate comparison plots for the error metrics across all run names for each
    deployment stage.
    """
    try:
        df = logger.load_all_logs(run_id=0)
    except FileNotFoundError as exc:
        print(f"Unable to load logs for multi-run deployment plots: {exc}")
        return

    if df.empty:
        print("No log data available for multi-run deployment plots.")
        return

    if "stage" not in df.columns:
        print("Column 'stage' not found in DataFrame. Cannot generate multi-run plots.")
        return

    if "run_name" not in df.columns:
        print(
            "Column 'run_name' not found in DataFrame. Cannot generate multi-run plots."
        )
        return

    for stage_name in DEPLOYMENT_STAGES:
        stage_data = df[df["stage"] == stage_name]
        if stage_data.empty:
            continue
        for metric in ERROR_METRICS:
            if metric not in stage_data.columns:
                continue
            plot_metric_multi_run(
                stage_data,
                metric,
                RESULTS_DIR,
                stage_name,
                "Time (s)",
                x_col="timestamp",
                to_seconds=True,
            )


def _resolve_column_name(df: pd.DataFrame, column: str | None) -> str | None:
    """
    Return the actual column name in the DataFrame that matches the requested column,
    falling back to a case-insensitive lookup when needed.
    """
    if column is None:
        return None
    if column in df.columns:
        return column
    column_map = {col.lower(): col for col in df.columns}
    return column_map.get(column.lower())


def _prepare_x_data(series: pd.Series, *, to_seconds: bool) -> pd.Series:
    """
    Convert the series to a numeric representation suitable for plotting.
    When ``to_seconds`` is True, convert datetime or timedelta data to seconds.
    """
    if series.empty:
        return series

    if to_seconds and pd.api.types.is_datetime64_any_dtype(series):
        base_time = series.iloc[0]
        return (series - base_time).dt.total_seconds()

    if to_seconds and pd.api.types.is_timedelta64_dtype(series):
        return series.dt.total_seconds()

    numeric_series = pd.to_numeric(series, errors="coerce")

    if to_seconds:
        return numeric_series.astype(float)

    return numeric_series


def main():
    """
    Main function to run the plotting script.
    """
    # Create logger
    logger = Logger()

    for run_name in [
        "pd_controller",
        "nn_controller",
        "nn_controller_adaptive",
        "ppo_nominal",
        "ppo_plain",
        "ppo_adaptive_sim",
        "ppo_adaptive_cnn",
        "ppo_adaptive_transformer",
        "pretrained_ppo_nominal",
        "pretrained_ppo",
        "pretrained_ppo_adaptive_sim",
        "pretrained_ppo_adaptive_cnn",
        "pretrained_ppo_adaptive_transformer",
    ]:
        plot_results(logger, run_name)

    plot_deployment_errors_all_runs(logger)


if __name__ == "__main__":
    main()
