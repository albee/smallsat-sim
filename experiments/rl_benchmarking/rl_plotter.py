import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from smallsat_sim.utils.logger import Logger


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "experiments" / "rl_results"

ERROR_METRICS = [
    "mean_lateral_error",
    "mean_angle_error",
    "mean_extrinsic_error",
]

STAGE_METRICS_TRAIN = {
    "policy_training": [
        "mean_lateral_error",
        "mean_angle_error",
        "mean_extrinsic_error",
        "mean_final_position_error",
        "mean_episodic_returns",
        "actor_loss_mean",
        "critic_loss_mean",
        "success_rate",
        "terminal_env_rate_at_end",
        "mean_std",
        "true_kl_mean",
        "clip_fraction",
        "explained_variance",
    ],
    "am_training": [
        "mean_lateral_error",
        "mean_angle_error",
        "mean_extrinsic_error",
        "am_train_loss_last",
        "am_train_loss_mean",
        "am_val_loss_last",
        "am_val_loss_mean",
    ],
    "evaluation": [
        "mean_lateral_error",
        "mean_angle_error",
        "mean_extrinsic_error",
        "mean_final_position_error",
        "mean_episodic_returns",
        "success_rate",
        "terminal_env_rate_at_end",
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
TRAJECTORY_MAX_TRACES = 100
# Set to an integer to restrict plotting to one execution (RunID).
# Keep as None to include all RunIDs.
RUN_ID_FILTER: int | None = None


def _parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the plotting script.
    """
    parser = argparse.ArgumentParser(description="Plot RL benchmarking results.")
    parser.add_argument(
        "-r",
        "--run-id",
        type=int,
        default=None,
        help="Filter logs by RunID. Omit to include all RunIDs.",
    )
    return parser.parse_args()


def plot_results(logger: Logger, run_name: str) -> None:
    """
    Plot the results of a simulation run (based on the logged data).
    """
    # Load **all** the logs
    df = logger.load_all_logs(run_id=RUN_ID_FILTER)

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

        plot_deployment_metrics_all_stages_for_run(df_run, dir_name, run_name)
        plot_deployment_trajectories(df_run, dir_name, run_name)


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
        df = logger.load_all_logs(run_id=RUN_ID_FILTER)
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


def plot_deployment_metrics_all_stages_for_run(
    df_run: pd.DataFrame,
    base_dir: Path,
    run_name: str,
) -> None:
    """
    For one run, compare deployment stages on the same axes for each deployment error metric.
    """
    if "stage" not in df_run.columns:
        return

    output_dir = base_dir / run_name / "deployment_all_stages"
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric in ERROR_METRICS:
        if metric not in df_run.columns:
            continue

        plt.figure(figsize=(9, 6))
        plotted = False

        for stage_name in DEPLOYMENT_STAGES:
            stage_data = df_run[df_run["stage"] == stage_name]
            if stage_data.empty:
                continue

            time_col = _resolve_column_name(stage_data, "timestamp")
            if time_col is None:
                time_col = _resolve_column_name(stage_data, "step")

            columns = [metric]
            if time_col is not None:
                columns.append(time_col)

            metric_df = stage_data[columns].dropna()
            if metric_df.empty:
                continue

            if time_col is not None:
                metric_df = metric_df.sort_values(time_col)
                x_series = _prepare_x_data(metric_df[time_col], to_seconds=True)
            else:
                x_series = pd.Series(
                    np.arange(len(metric_df), dtype=float), index=metric_df.index
                )

            y_series = metric_df[metric]
            valid_mask = x_series.notna()
            x_series = x_series[valid_mask]
            y_series = y_series[valid_mask]
            if y_series.empty:
                continue

            plt.plot(x_series, y_series, label=stage_name.replace("_", " "))
            plotted = True

        if not plotted:
            plt.close()
            continue

        metric_label = metric.replace("_", " ").title()
        plt.xlabel("Time (s)")
        plt.ylabel(metric_label)
        plt.title(f"{metric_label} Across Deployment Stages ({run_name})")
        plt.legend()
        plt.savefig(output_dir / f"{metric}_all_stages.svg", format="svg")
        plt.savefig(output_dir / f"{metric}_all_stages.pdf", format="pdf")
        plt.close()


def plot_deployment_trajectories(
    df_run: pd.DataFrame,
    base_dir: Path,
    run_name: str,
) -> None:
    """
    Plot 3D trajectories for each deployment stage and one combined deployment plot.
    """
    combined = {}
    any_stage_plotted = False

    for stage_name in DEPLOYMENT_STAGES:
        stage_df = df_run[df_run["stage"] == stage_name]
        trajectories = _extract_stage_trajectories(stage_df)
        if trajectories is None:
            continue

        any_stage_plotted = True
        combined[stage_name] = trajectories
        stage_output = base_dir / run_name / stage_name
        stage_output.mkdir(parents=True, exist_ok=True)
        _plot_3d_trajectories(
            trajectories,
            output_path=stage_output / "trajectories_3d",
            title=f"3D Trajectories ({stage_name.replace('_', ' ').title()})",
            color=None,
            label_prefix="sat",
        )

    if not any_stage_plotted:
        print(
            f"No deployment trajectory data found for '{run_name}'. "
            "Run deployment with the updated logger to generate 3D plots."
        )
        return

    _plot_3d_trajectories_all_stages(
        combined,
        output_path=base_dir / run_name / "deployment" / "trajectories_3d_all_stages",
    )


def _extract_stage_trajectories(
    stage_df: pd.DataFrame,
) -> np.ndarray | None:
    """
    Convert per-step logged vectorized positions into a ``[T, N, 3]`` trajectory tensor.
    """
    if stage_df.empty:
        return None

    pos_x_col = _resolve_column_name(stage_df, "position_x")
    pos_y_col = _resolve_column_name(stage_df, "position_y")
    pos_z_col = _resolve_column_name(stage_df, "position_z")
    if pos_x_col is None or pos_y_col is None or pos_z_col is None:
        return None

    order_col = _resolve_column_name(stage_df, "step")
    if order_col is None:
        order_col = _resolve_column_name(stage_df, "timestamp")

    if order_col is not None:
        data = stage_df.sort_values(order_col)
    else:
        data = stage_df

    x_entries, y_entries, z_entries = [], [], []
    for _, row in data[[pos_x_col, pos_y_col, pos_z_col]].dropna().iterrows():
        x = np.asarray(row[pos_x_col]).reshape(-1)
        y = np.asarray(row[pos_y_col]).reshape(-1)
        z = np.asarray(row[pos_z_col]).reshape(-1)
        if x.size == 0 or y.size == 0 or z.size == 0:
            continue
        if not (x.size == y.size == z.size):
            continue
        x_entries.append(x)
        y_entries.append(y)
        z_entries.append(z)

    if not x_entries:
        return None

    num_envs = min(arr.size for arr in x_entries)
    if num_envs < 1:
        return None

    x_mat = np.stack([arr[:num_envs] for arr in x_entries], axis=0)
    y_mat = np.stack([arr[:num_envs] for arr in y_entries], axis=0)
    z_mat = np.stack([arr[:num_envs] for arr in z_entries], axis=0)
    return np.stack([x_mat, y_mat, z_mat], axis=2)


def _plot_3d_trajectories(
    trajectories: np.ndarray,
    output_path: Path,
    title: str,
    color: str | None,
    label_prefix: str,
) -> None:
    """
    Plot a 3D trajectory figure from a ``[T, N, 3]`` trajectory tensor.
    """
    if trajectories.ndim != 3 or trajectories.shape[2] != 3:
        return

    timesteps, num_envs, _ = trajectories.shape
    if timesteps < 1 or num_envs < 1:
        return

    env_indices = _downsample_trace_indices(num_envs, TRAJECTORY_MAX_TRACES)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for env_idx in env_indices:
        x = trajectories[:, env_idx, 0]
        y = trajectories[:, env_idx, 1]
        z = trajectories[:, env_idx, 2]
        line_kwargs = {"alpha": 0.35, "linewidth": 1.0}
        if color is not None:
            line_kwargs["color"] = color
        ax.plot(x, y, z, **line_kwargs)
        ax.scatter(x[0], y[0], z[0], s=8, color=line_kwargs.get("color", "tab:blue"))

    # Mark the final point of the first plotted satellite for orientation.
    first_idx = int(env_indices[0])
    ax.scatter(
        trajectories[-1, first_idx, 0],
        trajectories[-1, first_idx, 1],
        trajectories[-1, first_idx, 2],
        s=40,
        marker="x",
        color="black",
        label=f"{label_prefix} end",
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(
        f"{title}\nshowing {len(env_indices)} of {num_envs} trajectories"
    )
    ax.legend(loc="upper right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output_path}.svg", format="svg")
    fig.savefig(f"{output_path}.pdf", format="pdf")
    plt.close(fig)


def _plot_3d_trajectories_all_stages(
    stage_trajectories: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """
    Plot all deployment stages in a single 3D figure.
    """
    if not stage_trajectories:
        return

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("tab10")
    color_idx = 0

    for stage_name in DEPLOYMENT_STAGES:
        if stage_name not in stage_trajectories:
            continue
        trajectories = stage_trajectories[stage_name]
        _, num_envs, _ = trajectories.shape
        env_indices = _downsample_trace_indices(num_envs, TRAJECTORY_MAX_TRACES // 2)
        stage_color = cmap(color_idx % 10)
        color_idx += 1
        label = stage_name.replace("_", " ")

        for env_idx in env_indices:
            ax.plot(
                trajectories[:, env_idx, 0],
                trajectories[:, env_idx, 1],
                trajectories[:, env_idx, 2],
                color=stage_color,
                alpha=0.2,
                linewidth=0.8,
            )

        first_idx = int(env_indices[0])
        ax.plot(
            [trajectories[0, first_idx, 0]],
            [trajectories[0, first_idx, 1]],
            [trajectories[0, first_idx, 2]],
            marker="o",
            markersize=4,
            color=stage_color,
            linestyle="None",
            label=label,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Trajectories Across Deployment Stages")
    ax.legend(loc="upper right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output_path}.svg", format="svg")
    fig.savefig(f"{output_path}.pdf", format="pdf")
    plt.close(fig)


def _downsample_trace_indices(num_envs: int, max_traces: int) -> np.ndarray:
    """
    Return evenly-spaced trajectory indices capped by ``max_traces``.
    """
    if num_envs <= max_traces:
        return np.arange(num_envs, dtype=int)
    return np.linspace(0, num_envs - 1, num=max_traces, dtype=int)


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
    global RUN_ID_FILTER
    args = _parse_args()
    RUN_ID_FILTER = args.run_id

    # Create logger
    logger = Logger()

    for run_name in [
        "pd_controller",
        "nominal_mpc",
        "lqr",
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
