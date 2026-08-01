"""Batch-first command-line interface."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Sequence

from .core import ExtrusionOptions, extrude_plan
from .exporters import export_plan
from .loader import load_dataset, load_splits, select_plans


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="resplan-extrude",
        description="Extrude ResPlan floor plans to metric OBJ or GLB models.",
    )
    parser.add_argument("--data", default="ResPlan.pkl", help="dataset pickle path")
    parser.add_argument("--splits", default="split.json", help="split JSON path")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--ids", help="comma-separated public plan IDs")
    selection.add_argument("--split", help="canonical split name")
    selection.add_argument(
        "--all", action="store_true", dest="all_plans", help="process every plan"
    )
    parser.add_argument("--limit", type=int, help="cap the selected plan count")
    parser.add_argument("--output", default="exports", help="output root directory")
    parser.add_argument(
        "--format", choices=("obj", "glb", "both"), default="obj"
    )
    parser.add_argument(
        "--on-error", choices=("skip", "fail"), default="skip"
    )
    parser.add_argument("--wall-thickness", type=float, default=0.20)
    parser.add_argument("--wall-height", type=float, default=2.70)
    parser.add_argument("--floor-thickness", type=float, default=0.20)
    parser.add_argument("--ceiling", action="store_true")
    parser.add_argument("--ceiling-thickness", type=float, default=0.15)
    parser.add_argument("--door-height", type=float, default=2.10)
    parser.add_argument(
        "--door-mode",
        choices=("lintel", "full-height"),
        default="lintel",
        help="retain a lintel or leave door openings clear to wall height",
    )
    parser.add_argument(
        "--close-boundary-doors",
        action="store_true",
        help=(
            "fill front, exterior, and balcony door footprints with "
            "full-height wall"
        ),
    )
    parser.add_argument(
        "--restricted-door-count",
        type=int,
        default=0,
        help=(
            "number of non-boundary interior doors given a separate "
            "height and/or width restriction"
        ),
    )
    parser.add_argument(
        "--restricted-door-mode",
        choices=("height", "width", "both"),
        default="height",
        help="restrict selected doors by clearance height, opening width, or both",
    )
    parser.add_argument(
        "--restricted-door-height",
        type=float,
        default=1.00,
        help="clearance height for height-restricted doors",
    )
    parser.add_argument(
        "--restricted-door-width",
        type=float,
        default=0.40,
        help="centred opening width in metres for width-restricted doors",
    )
    parser.add_argument(
        "--restricted-door-seed",
        type=int,
        default=0,
        help="reproducible seed used to choose restricted interior doors",
    )
    parser.add_argument("--window-sill-height", type=float, default=0.90)
    parser.add_argument("--window-head-height", type=float, default=2.10)
    parser.add_argument(
        "--window-mode",
        choices=("opening", "solid"),
        default="opening",
        help="model window openings or fill their footprints with wall",
    )
    parser.add_argument(
        "--exclude-balcony",
        action="store_false",
        dest="include_balcony",
        default=True,
    )
    parser.add_argument(
        "--no-center", action="store_false", dest="center", default=True
    )
    geometry = parser.add_argument_group("seeded geometry variations")
    geometry.add_argument(
        "--diagonal-corner-percent",
        type=float,
        default=0.0,
        help="percentage of eligible wall corners replaced by diagonals",
    )
    geometry.add_argument(
        "--diagonal-corner-size",
        type=float,
        default=0.30,
        help="diagonal corner cut depth in metres",
    )
    geometry.add_argument(
        "--rounded-corner-percent",
        type=float,
        default=0.0,
        help="percentage of eligible wall corners rounded",
    )
    geometry.add_argument(
        "--rounded-corner-radius",
        type=float,
        default=0.30,
        help="rounded corner radius in metres",
    )
    geometry.add_argument(
        "--curved-wall-percent",
        type=float,
        default=0.0,
        help="percentage of eligible wall-face segments bowed into curves",
    )
    geometry.add_argument(
        "--curved-wall-amplitude",
        type=float,
        default=0.15,
        help="maximum wall bow in metres",
    )
    geometry.add_argument(
        "--noisy-wall-percent",
        type=float,
        default=0.0,
        help="percentage of eligible wall-face segments made irregular",
    )
    geometry.add_argument(
        "--noisy-wall-amplitude",
        type=float,
        default=0.08,
        help="maximum seeded wall jitter in metres",
    )
    geometry.add_argument(
        "--geometry-seed",
        type=int,
        default=0,
        help="reproducible seed for every geometry variation",
    )
    return parser


def _options(args: argparse.Namespace) -> ExtrusionOptions:
    return ExtrusionOptions(
        wall_thickness=args.wall_thickness,
        wall_height=args.wall_height,
        floor_thickness=args.floor_thickness,
        ceiling=args.ceiling,
        ceiling_thickness=args.ceiling_thickness,
        door_height=args.door_height,
        door_mode=args.door_mode,
        close_boundary_doors=args.close_boundary_doors,
        restricted_door_count=args.restricted_door_count,
        restricted_door_mode=args.restricted_door_mode,
        restricted_door_height=args.restricted_door_height,
        restricted_door_width=args.restricted_door_width,
        restricted_door_seed=args.restricted_door_seed,
        window_sill_height=args.window_sill_height,
        window_head_height=args.window_head_height,
        window_mode=args.window_mode,
        include_balcony=args.include_balcony,
        center=args.center,
        diagonal_corner_percent=args.diagonal_corner_percent,
        diagonal_corner_size=args.diagonal_corner_size,
        rounded_corner_percent=args.rounded_corner_percent,
        rounded_corner_radius=args.rounded_corner_radius,
        curved_wall_percent=args.curved_wall_percent,
        curved_wall_amplitude=args.curved_wall_amplitude,
        noisy_wall_percent=args.noisy_wall_percent,
        noisy_wall_amplitude=args.noisy_wall_amplitude,
        geometry_seed=args.geometry_seed,
    )


def _write_failure(output: Path, plan_id: object, exc: Exception) -> str:
    plan_dir = output / str(plan_id)
    plan_dir.mkdir(parents=True, exist_ok=True)
    path = plan_dir / "metadata.json"
    path.write_text(
        json.dumps(
            {
                "plan_id": plan_id,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return str(path)


def run(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        options = _options(args)
        options.validate()
        plans = load_dataset(args.data)
        splits = load_splits(args.splits) if args.split else None
        ids = (
            [value.strip() for value in args.ids.split(",") if value.strip()]
            if args.ids
            else None
        )
        selected = select_plans(
            plans,
            ids=ids,
            split=args.split,
            all_plans=args.all_plans,
            splits=splits,
            limit=args.limit,
        )
    except Exception as exc:
        parser.error(str(exc))

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(Path(args.data)),
        "selected_count": len(selected),
        "format": args.format,
        "options": asdict(options),
        "successes": [],
        "failures": [],
    }

    for position, plan in enumerate(selected, start=1):
        plan_id = plan.get("id", f"index-{position - 1}")
        try:
            result = extrude_plan(plan, options)
            artifacts = export_plan(result, output, args.format)
            manifest["successes"].append(
                {
                    "plan_id": result.plan_id,
                    "artifacts": artifacts,
                    "warnings": result.warnings,
                }
            )
            print(f"[{position}/{len(selected)}] exported plan {result.plan_id}")
        except Exception as exc:
            metadata = _write_failure(output, plan_id, exc)
            failure = {
                "plan_id": plan_id,
                "error": f"{type(exc).__name__}: {exc}",
                "metadata": metadata,
            }
            manifest["failures"].append(failure)
            print(
                f"[{position}/{len(selected)}] failed plan {plan_id}: {exc}",
                file=sys.stderr,
            )
            if args.on_error == "fail":
                break

    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifest["success_count"] = len(manifest["successes"])
    manifest["failure_count"] = len(manifest["failures"])
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"manifest: {manifest_path}")

    success_count = int(manifest["success_count"])
    failure_count = int(manifest["failure_count"])
    if success_count == 0 or (len(selected) == 1 and failure_count):
        return 1
    if args.on_error == "fail" and failure_count:
        return 1
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
