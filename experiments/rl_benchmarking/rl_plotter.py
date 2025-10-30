from pathlib import Path
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

    # Define base output directory and keep runs grouped by stage
    dir_name = Path("experiments/rl_results/")

    # Define a mapping from stage names to the list of metrics to plot
    stage_metrics_train = {
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

    # If the stage column exists, loop over the stage-metrics mapping
    if "stage" in df_run.columns:
        for stage_name, metrics in stage_metrics_train.items():
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

    errors = ["mean_tracking_error", "mean_angle_error", "mean_extrinsic_error"]
    stage_metrics_test = {
        "deployment": errors,
        "stuck_off_deployment": errors,
        "stuck_on_deployment": errors,
        "faulty_valve_deployment": errors,
        "saturated_thrust_deployment": errors,
        "thrust_instability_deployment": errors,
        "constant_force_disturbances_deployment": errors,
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
        "ppo_adaptive",
        "pretrained_ppo_nominal",
        "pretrained_ppo",
        "pretrained_ppo_adaptive_sim",
        "pretrained_ppo_adaptive",
    ]:
        plot_results(logger, run_name)


if __name__ == "__main__":
    main()
