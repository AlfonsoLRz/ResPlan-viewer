"""Dataset loading and plan selection helpers."""

from __future__ import annotations

import json
from pathlib import Path
import pickle
from typing import Any, Iterable


def load_dataset(path: str | Path = "ResPlan.pkl") -> list[dict[str, Any]]:
    """Load a trusted ResPlan pickle file."""
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")
    with dataset_path.open("rb") as handle:
        plans = pickle.load(handle)
    if not isinstance(plans, list):
        raise ValueError("expected the dataset pickle to contain a list")
    return plans


def load_splits(path: str | Path = "split.json") -> dict[str, list[Any]]:
    split_path = Path(path)
    if not split_path.is_file():
        raise FileNotFoundError(f"split file not found: {split_path}")
    with split_path.open("r", encoding="utf-8") as handle:
        splits = json.load(handle)
    if not isinstance(splits, dict):
        raise ValueError("expected split JSON to contain an object")
    return splits


def _id_key(value: Any) -> str:
    return str(value)


def select_plans(
    plans: list[dict[str, Any]],
    *,
    ids: Iterable[int | str] | None = None,
    split: str | None = None,
    all_plans: bool = False,
    splits: dict[str, list[Any]] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Select plans by public dataset ID while preserving requested order."""
    selectors = sum((ids is not None, split is not None, all_plans))
    if selectors != 1:
        raise ValueError("select exactly one of ids, split, or all_plans")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be greater than zero")

    index = {_id_key(plan.get("id")): plan for plan in plans}
    if len(index) != len(plans):
        raise ValueError("dataset contains duplicate plan IDs")

    if ids is not None:
        requested = [_id_key(value) for value in ids]
    elif split is not None:
        if splits is None:
            raise ValueError("split selection requires loaded split metadata")
        if split not in splits:
            choices = ", ".join(sorted(splits))
            raise ValueError(f"unknown split '{split}'; choose from {choices}")
        requested = [_id_key(value) for value in splits[split]]
    else:
        selected = list(plans)
        return selected[:limit] if limit is not None else selected

    missing = [value for value in requested if value not in index]
    if missing:
        preview = ", ".join(missing[:10])
        suffix = "..." if len(missing) > 10 else ""
        raise KeyError(f"plan IDs not found: {preview}{suffix}")
    selected = [index[value] for value in requested]
    return selected[:limit] if limit is not None else selected
