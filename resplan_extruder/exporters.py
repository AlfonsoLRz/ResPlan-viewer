"""OBJ/GLB serialization and per-plan output folders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .core import ExtrusionResult


VALID_FORMATS = {"obj", "glb"}


def _normalize_formats(formats: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(formats, str):
        values = ("obj", "glb") if formats == "both" else (formats,)
    else:
        values = tuple(formats)
    unknown = set(values).difference(VALID_FORMATS)
    if not values or unknown:
        raise ValueError(f"formats must contain only {sorted(VALID_FORMATS)}")
    return values


def _obj_text(result: ExtrusionResult) -> str:
    lines = [
        "# ResPlan floor-plan extrusion",
        f"# plan_id: {result.plan_id}",
        "# units: metres; up axis: Z",
    ]
    vertex_offset = 1
    for name, mesh in result.components.items():
        lines.append(f"o {name}")
        lines.append(f"g {name}")
        for vertex in np.asarray(mesh.vertices):
            lines.append(f"v {vertex[0]:.9g} {vertex[1]:.9g} {vertex[2]:.9g}")
        for face in np.asarray(mesh.faces):
            a, b, c = (int(value) + vertex_offset for value in face)
            lines.append(f"f {a} {b} {c}")
        vertex_offset += len(mesh.vertices)
    return "\n".join(lines) + "\n"


def export_bytes(result: ExtrusionResult, format: str = "obj") -> bytes:
    """Serialize one result without writing files."""
    if format == "obj":
        return _obj_text(result).encode("utf-8")
    if format == "glb":
        payload = result.scene().export(file_type="glb")
        if not isinstance(payload, bytes):
            payload = bytes(payload)
        return payload
    raise ValueError(f"unsupported format: {format}")


def export_plan(
    result: ExtrusionResult,
    output_dir: str | Path,
    formats: str | Iterable[str] = "obj",
) -> dict[str, str]:
    """Write model files and metadata into ``output_dir/<plan_id>``."""
    selected_formats = _normalize_formats(formats)
    plan_dir = Path(output_dir) / str(result.plan_id)
    plan_dir.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, str] = {}
    for format in selected_formats:
        path = plan_dir / f"plan.{format}"
        path.write_bytes(export_bytes(result, format))
        artifacts[format] = str(path)

    metadata_path = plan_dir / "metadata.json"
    metadata = result.metadata()
    metadata["artifacts"] = artifacts
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    artifacts["metadata"] = str(metadata_path)
    return artifacts
