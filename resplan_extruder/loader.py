"""Dataset loading and plan selection helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import pickle
import tempfile
from typing import Any, Iterable
from urllib.request import Request, urlopen


DEFAULT_DATASET_URL = (
    "https://github.com/AlfonsoLRz/ResPlan-viewer/releases/latest/"
    "download/ResPlan.pkl"
)
DEFAULT_DATASET_SHA256 = (
    "2a73179cf11e6066384400494683072eb1648ed56ae000750d0e9f3fa499c570"
)


def ensure_dataset(
    path: str | Path,
    *,
    url: str | None = None,
    sha256: str | None = None,
) -> Path:
    """Return a local dataset, downloading and verifying it when absent."""
    dataset_path = Path(path).expanduser()
    if dataset_path.is_file():
        return dataset_path
    if not url:
        raise FileNotFoundError(
            f"dataset not found: {dataset_path}; no download URL was configured"
        )

    expected_hash = sha256.strip().lower() if sha256 else None
    if expected_hash and (
        len(expected_hash) != 64
        or any(
            character not in "0123456789abcdef"
            for character in expected_hash
        )
    ):
        raise ValueError("dataset SHA-256 must contain exactly 64 hexadecimal digits")

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    digest = hashlib.sha256()
    try:
        request = Request(url, headers={"User-Agent": "resplan-extruder/0.1"})
        with urlopen(request, timeout=120) as response, tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{dataset_path.name}.",
            suffix=".download",
            dir=dataset_path.parent,
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
            output.flush()
            os.fsync(output.fileno())

        actual_hash = digest.hexdigest()
        if expected_hash and actual_hash != expected_hash:
            raise ValueError(
                "dataset checksum mismatch: "
                f"expected {expected_hash}, downloaded {actual_hash}"
            )
        os.replace(temporary_path, dataset_path)
        temporary_path = None
        return dataset_path
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


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
