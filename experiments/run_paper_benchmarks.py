import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass
class AggregateRow:
    experiment: str
    controller: str
    failure_mode: str
    task_success_rate: float | None
    lateral_error: float | None
    control_effort: float | None
    rendezvous_success_rate: float | None
    docking_success_rate: float | None
    final_position_error: float | None
    final_attitude_error: float | None
    max_contact_force: float | None
    result_dir: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run paper-ready benchmark suites for inspection and docking, then build "
            "a consolidated results table."
        )
    )
    parser.add_argument("--trials-inspection", type=int, default=6)
    parser.add_argument("--trials-docking", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--failure-time", type=float, default=30.0)
    parser.add_argument(
        "--dock-surface-offset",
        type=float,
        default=0.0,
        help=(
            "Shift docking terminal target along -approach_axis [m]. "
            "Positive values move target inward toward station surface."
        ),
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default="experiments/results/paper_benchmarks",
    )
    parser.add_argument(
        "--skip-inspection",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--skip-docking",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--save-snapshots",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print commands without executing.",
    )
    return parser.parse_args()


def _run_command(cmd: list[str], *, dry_run: bool) -> None:
    print("[run]", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _inspection_rows(
    summary_json: dict[str, Any],
    *,
    failure_mode: str,
    result_dir: Path,
) -> list[AggregateRow]:
    aggregate = summary_json.get("aggregate", {})
    rows: list[AggregateRow] = []
    for controller, metrics in aggregate.items():
        rows.append(
            AggregateRow(
                experiment="inspection",
                controller=str(controller),
                failure_mode=failure_mode,
                task_success_rate=_safe_float(metrics.get("task_success_rate")),
                lateral_error=_safe_float(metrics.get("mean_lateral_tracking_error")),
                control_effort=_safe_float(metrics.get("mean_control_effort")),
                rendezvous_success_rate=None,
                docking_success_rate=None,
                final_position_error=None,
                final_attitude_error=None,
                max_contact_force=None,
                result_dir=str(result_dir),
            )
        )
    return rows


def _docking_row(summary_json: dict[str, Any], *, result_dir: Path) -> AggregateRow:
    aggregate = summary_json.get("aggregate", {})
    return AggregateRow(
        experiment="docking_surrogate",
        controller="nominal_mpc",
        failure_mode="soft_contact",
        task_success_rate=None,
        lateral_error=None,
        control_effort=_safe_float(aggregate.get("mean_control_effort")),
        rendezvous_success_rate=_safe_float(aggregate.get("rendezvous_success_rate")),
        docking_success_rate=_safe_float(aggregate.get("docking_success_rate")),
        final_position_error=_safe_float(aggregate.get("mean_final_position_error")),
        final_attitude_error=_safe_float(aggregate.get("mean_final_attitude_error")),
        max_contact_force=_safe_float(aggregate.get("max_contact_force_observed")),
        result_dir=str(result_dir),
    )


def _write_csv(rows: list[AggregateRow], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(asdict(rows[0]).keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _fmt(x: float | None, digits: int = 4) -> str:
    if x is None:
        return "-"
    return f"{x:.{digits}f}"


def _write_markdown(rows: list[AggregateRow], path: Path) -> None:
    header = (
        "| Experiment | Controller | Failure | Success | Lateral Error | "
        "Control Effort | Rendezvous | Docking | Final Pos Err | Final Att Err | "
        "Max Contact Force | Result Dir |"
    )
    sep = "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"
    lines = [header, sep]
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    r.experiment,
                    r.controller,
                    r.failure_mode,
                    _fmt(r.task_success_rate),
                    _fmt(r.lateral_error),
                    _fmt(r.control_effort),
                    _fmt(r.rendezvous_success_rate),
                    _fmt(r.docking_success_rate),
                    _fmt(r.final_position_error),
                    _fmt(r.final_attitude_error),
                    _fmt(r.max_contact_force),
                    r.result_dir,
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    out_root = (repo_root / args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    python = sys.executable
    rows: list[AggregateRow] = []

    if not args.skip_inspection:
        for mode in ("nominal", "stuck_off", "stuck_on"):
            out_dir = out_root / f"inspection_{mode}"
            cmd = [
                python,
                str((repo_root / "experiments" / "inspection_benchmark.py").resolve()),
                "--controllers",
                "pd,nominal_mpc",
                "--trials",
                str(args.trials_inspection),
                "--seed",
                str(args.seed),
                "--failure-mode",
                mode,
                "--failure-time",
                str(args.failure_time),
                "--out-dir",
                str(out_dir),
                "--headless" if args.headless else "--no-headless",
            ]
            _run_command(cmd, dry_run=args.dry_run)
            if not args.dry_run:
                summary = _read_json(out_dir / "summary.json")
                rows.extend(_inspection_rows(summary, failure_mode=mode, result_dir=out_dir))

    if not args.skip_docking:
        out_dir = out_root / "docking_surrogate_contact"
        cmd = [
            python,
            str((repo_root / "experiments" / "docking_surrogate_contact.py").resolve()),
            "--trials",
            str(args.trials_docking),
            "--seed",
            str(args.seed),
            "--out-dir",
            str(out_dir),
            "--dock-surface-offset",
            str(args.dock_surface_offset),
            "--headless" if args.headless else "--no-headless",
            "--save-snapshots" if args.save_snapshots else "--no-save-snapshots",
        ]
        _run_command(cmd, dry_run=args.dry_run)
        if not args.dry_run:
            summary = _read_json(out_dir / "summary.json")
            rows.append(_docking_row(summary, result_dir=out_dir))

    if args.dry_run:
        print("[done] dry-run complete (no files consolidated).")
        return

    consolidated_json = out_root / "consolidated_summary.json"
    consolidated_csv = out_root / "consolidated_table.csv"
    consolidated_md = out_root / "consolidated_table.md"

    payload = {
        "config": {
            "trials_inspection": int(args.trials_inspection),
            "trials_docking": int(args.trials_docking),
            "seed": int(args.seed),
            "failure_time": float(args.failure_time),
            "dock_surface_offset": float(args.dock_surface_offset),
            "headless": bool(args.headless),
            "save_snapshots": bool(args.save_snapshots),
            "out_root": str(out_root),
        },
        "rows": [asdict(r) for r in rows],
    }
    consolidated_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(rows, consolidated_csv)
    _write_markdown(rows, consolidated_md)

    print("[done] benchmark suite complete")
    print(f"[done] consolidated json: {consolidated_json}")
    print(f"[done] consolidated csv:  {consolidated_csv}")
    print(f"[done] consolidated md:   {consolidated_md}")


if __name__ == "__main__":
    main()
