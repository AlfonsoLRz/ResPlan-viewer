from __future__ import annotations

import json
import pickle

from resplan_extruder.cli import run
from resplan_extruder.loader import select_plans

from .test_core import synthetic_plan


def test_select_by_ids_split_all_and_limit() -> None:
    plans = [{"id": 3}, {"id": 1}, {"id": 2}]
    assert [p["id"] for p in select_plans(plans, ids=[2, 3])] == [2, 3]
    assert [
        p["id"]
        for p in select_plans(
            plans, split="test", splits={"test": [1, 3]}
        )
    ] == [1, 3]
    assert [
        p["id"] for p in select_plans(plans, all_plans=True, limit=2)
    ] == [3, 1]


def test_cli_single_and_both_formats(tmp_path) -> None:
    data_path = tmp_path / "plans.pkl"
    split_path = tmp_path / "split.json"
    output = tmp_path / "exports"
    with data_path.open("wb") as handle:
        pickle.dump([synthetic_plan()], handle)
    split_path.write_text(json.dumps({"test": [42]}), encoding="utf-8")

    code = run(
        [
            "--data",
            str(data_path),
            "--splits",
            str(split_path),
            "--split",
            "test",
            "--output",
            str(output),
            "--format",
            "both",
            "--ceiling",
            "--door-mode",
            "full-height",
            "--restricted-door-count",
            "1",
            "--restricted-door-height",
            "0.75",
            "--restricted-door-seed",
            "17",
            "--window-mode",
            "solid",
            "--diagonal-corner-percent",
            "25",
            "--rounded-corner-percent",
            "25",
            "--curved-wall-percent",
            "25",
            "--noisy-wall-percent",
            "25",
            "--geometry-seed",
            "23",
        ]
    )
    assert code == 0
    assert (output / "42" / "plan.obj").is_file()
    assert (output / "42" / "plan.glb").is_file()
    assert (output / "42" / "metadata.json").is_file()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["success_count"] == 1
    assert manifest["failure_count"] == 0
    metadata = json.loads(
        (output / "42" / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["options"]["door_mode"] == "full-height"
    assert metadata["options"]["restricted_door_count"] == 1
    assert metadata["options"]["restricted_door_height"] == 0.75
    assert metadata["options"]["restricted_door_seed"] == 17
    assert metadata["options"]["window_mode"] == "solid"
    assert metadata["options"]["diagonal_corner_percent"] == 25
    assert metadata["options"]["rounded_corner_percent"] == 25
    assert metadata["options"]["curved_wall_percent"] == 25
    assert metadata["options"]["noisy_wall_percent"] == 25
    assert metadata["options"]["geometry_seed"] == 23
    assert metadata["geometry_variations"]["enabled"] is True
    assert "door_lintels" not in metadata["components"]
    assert "restricted_door_lintels" in metadata["components"]
    assert "window_sills" not in metadata["components"]


def test_cli_single_failure_is_nonzero(tmp_path) -> None:
    broken = synthetic_plan()
    broken["wall_depth"] = 0
    data_path = tmp_path / "plans.pkl"
    with data_path.open("wb") as handle:
        pickle.dump([broken], handle)

    code = run(
        [
            "--data",
            str(data_path),
            "--ids",
            "42",
            "--output",
            str(tmp_path / "exports"),
        ]
    )
    assert code == 1
