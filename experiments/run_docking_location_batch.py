import argparse
import subprocess
import sys
from pathlib import Path


# Candidate docking locations inferred from Gateway mesh regions.
DOCK_SITES = [
    "dock_orion_interface_a",
    "dock_halo_ihab_interface_a",
    "dock_halo_ppe_interface_a",
    "dock_crew_airlock_a",
    "dock_hls_side_a",
    "dock_logistics_side_a",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run docking_surrogate_contact.py once per candidate dock location, "
            "saving each run into a separate output folder."
        )
    )
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dock-surface-offset", type=float, default=1.0)
    parser.add_argument("--post-contact-timeout", type=float, default=10.0)
    parser.add_argument("--vehicle", type=str, choices=("astrobee", "cubesat"), default="astrobee")
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to run docking simulation headless.",
    )
    parser.add_argument(
        "--save-snapshots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to save docking snapshots.",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default="experiments/results/docking_location_batch",
        help="Root directory under which per-site folders are created.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    out_root = (repo_root / args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    script_path = repo_root / "experiments" / "docking_surrogate_contact.py"

    for dock_site in DOCK_SITES:
        site_out = out_root / dock_site
        site_out.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(script_path),
            "--trials",
            str(args.trials),
            "--seed",
            str(args.seed),
            "--vehicle",
            str(args.vehicle),
            "--dock-site",
            dock_site,
            "--dock-surface-offset",
            str(args.dock_surface_offset),
            "--post-contact-timeout",
            str(args.post_contact_timeout),
            "--out-dir",
            str(site_out),
        ]

        cmd.append("--headless" if args.headless else "--no-headless")
        cmd.append("--save-snapshots" if args.save_snapshots else "--no-save-snapshots")

        print(f"[dock-batch] running site={dock_site}")
        print(f"[dock-batch] out={site_out}")
        subprocess.run(cmd, check=True, cwd=str(repo_root))

    print("[dock-batch] complete")
    print(f"[dock-batch] outputs root: {out_root}")


if __name__ == "__main__":
    main()
