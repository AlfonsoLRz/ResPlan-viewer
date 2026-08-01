"""Geometry cleanup, scaling, and extrusion for ResPlan records."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import math
import random
from typing import Any, Iterable

import numpy as np
from shapely import affinity, make_valid, set_precision
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiPolygon,
    Point,
    Polygon,
    box,
)
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
import trimesh


COMPONENT_COLORS: dict[str, tuple[int, int, int, int]] = {
    "floor": (178, 178, 178, 255),
    "walls": (225, 220, 205, 255),
    "door_lintels": (205, 190, 165, 255),
    "restricted_door_lintels": (190, 105, 75, 255),
    "window_sills": (190, 205, 215, 255),
    "window_headers": (190, 205, 215, 255),
    "ceiling": (235, 235, 235, 210),
}

GEOMETRY_GRID_SIZE = 1e-9
PLAN_SCALE_REFERENCE_WALL_THICKNESS = 0.20
FLOOR_SPACE_KEYS = (
    "living",
    "bedroom",
    "bathroom",
    "kitchen",
    "storage",
    "stair",
)


@dataclass(frozen=True)
class ExtrusionOptions:
    """Metric dimensions and inclusion flags for one extrusion."""

    wall_thickness: float = 0.20
    wall_height: float = 2.70
    floor_thickness: float = 0.20
    ceiling: bool = False
    ceiling_thickness: float = 0.15
    door_height: float = 2.10
    door_mode: str = "lintel"
    close_boundary_doors: bool = False
    restricted_door_count: int = 0
    restricted_door_mode: str = "height"
    restricted_door_height: float = 1.00
    restricted_door_width: float = 0.40
    restricted_door_seed: int = 0
    window_sill_height: float = 0.90
    window_head_height: float = 2.10
    window_mode: str = "opening"
    include_balcony: bool = True
    center: bool = True
    diagonal_corner_percent: float = 0.0
    diagonal_corner_size: float = 0.30
    rounded_corner_percent: float = 0.0
    rounded_corner_radius: float = 0.30
    curved_wall_percent: float = 0.0
    curved_wall_amplitude: float = 0.15
    noisy_wall_percent: float = 0.0
    noisy_wall_amplitude: float = 0.08
    geometry_seed: int = 0

    def validate(self) -> None:
        positive = {
            "wall_thickness": self.wall_thickness,
            "wall_height": self.wall_height,
            "floor_thickness": self.floor_thickness,
            "ceiling_thickness": self.ceiling_thickness,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite number greater than zero")
        percentages = {
            "diagonal_corner_percent": self.diagonal_corner_percent,
            "rounded_corner_percent": self.rounded_corner_percent,
            "curved_wall_percent": self.curved_wall_percent,
            "noisy_wall_percent": self.noisy_wall_percent,
        }
        for name, value in percentages.items():
            if not math.isfinite(value) or not 0 <= value <= 100:
                raise ValueError(f"{name} must be between 0 and 100")
        variation_sizes = {
            "diagonal_corner_size": self.diagonal_corner_size,
            "rounded_corner_radius": self.rounded_corner_radius,
            "curved_wall_amplitude": self.curved_wall_amplitude,
            "noisy_wall_amplitude": self.noisy_wall_amplitude,
        }
        for name, value in variation_sizes.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite number >= 0")
        active_size_pairs = (
            (self.diagonal_corner_percent, self.diagonal_corner_size, "diagonal_corner_size"),
            (self.rounded_corner_percent, self.rounded_corner_radius, "rounded_corner_radius"),
            (self.curved_wall_percent, self.curved_wall_amplitude, "curved_wall_amplitude"),
            (self.noisy_wall_percent, self.noisy_wall_amplitude, "noisy_wall_amplitude"),
        )
        for percent, value, name in active_size_pairs:
            if percent > 0 and value <= 0:
                raise ValueError(f"{name} must be greater than zero when enabled")
        if self.door_mode not in {"lintel", "full-height"}:
            raise ValueError("door_mode must be 'lintel' or 'full-height'")
        if self.window_mode not in {"opening", "solid"}:
            raise ValueError("window_mode must be 'opening' or 'solid'")
        if self.restricted_door_mode not in {"height", "width", "both"}:
            raise ValueError(
                "restricted_door_mode must be 'height', 'width', or 'both'"
            )
        if (
            isinstance(self.restricted_door_count, bool)
            or not isinstance(self.restricted_door_count, (int, np.integer))
            or self.restricted_door_count < 0
        ):
            raise ValueError("restricted_door_count must be an integer >= 0")
        if (
            isinstance(self.restricted_door_seed, bool)
            or not isinstance(self.restricted_door_seed, (int, np.integer))
            or self.restricted_door_seed < 0
        ):
            raise ValueError("restricted_door_seed must be an integer >= 0")
        if (
            isinstance(self.geometry_seed, bool)
            or not isinstance(self.geometry_seed, (int, np.integer))
            or self.geometry_seed < 0
        ):
            raise ValueError("geometry_seed must be an integer >= 0")
        if (
            self.restricted_door_count > 0
            and self.restricted_door_mode in {"height", "both"}
            and (
                not math.isfinite(self.restricted_door_height)
                or not 0 <= self.restricted_door_height < self.wall_height
            )
        ):
            raise ValueError(
                "restricted_door_height must satisfy "
                "0 <= restricted_door_height < wall_height"
            )
        if (
            self.restricted_door_count > 0
            and self.restricted_door_mode in {"width", "both"}
            and (
                not math.isfinite(self.restricted_door_width)
                or self.restricted_door_width <= 0
            )
        ):
            raise ValueError(
                "restricted_door_width must be a finite number greater than zero"
            )
        if self.door_mode == "lintel" and (
            not math.isfinite(self.door_height)
            or not 0 < self.door_height < self.wall_height
        ):
            raise ValueError("door_height must satisfy 0 < door_height < wall_height")
        if self.window_mode == "opening" and (
            not math.isfinite(self.window_sill_height)
            or not math.isfinite(self.window_head_height)
            or self.window_sill_height < 0
            or self.window_sill_height >= self.window_head_height
            or self.window_head_height >= self.wall_height
        ):
            raise ValueError(
                "window heights must satisfy 0 <= sill < head < wall_height"
            )


@dataclass
class ExtrusionResult:
    """Meshes and provenance for one extruded plan."""

    plan_id: int | str
    components: dict[str, trimesh.Trimesh]
    scale_factor: float
    source_wall_depth: float
    source_center: tuple[float, float]
    dimensions: tuple[float, float, float]
    options: ExtrusionOptions
    warnings: list[str] = field(default_factory=list)
    door_treatments: dict[str, Any] = field(default_factory=dict)
    geometry_variations: dict[str, Any] = field(default_factory=dict)

    @property
    def mesh(self) -> trimesh.Trimesh:
        """Return all semantic components as one mesh."""
        if not self.components:
            raise ValueError("extrusion produced no mesh components")
        return trimesh.util.concatenate(tuple(self.components.values()))

    def scene(self, include_ceiling: bool = True) -> trimesh.Scene:
        """Return a scene retaining semantic component names and colors."""
        scene = trimesh.Scene()
        for name, mesh in self.components.items():
            if name == "ceiling" and not include_ceiling:
                continue
            scene.add_geometry(mesh.copy(), geom_name=name, node_name=name)
        return scene

    def metadata(self, status: str = "success") -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "status": status,
            "scale_factor_m_per_source_unit": self.scale_factor,
            "plan_scale_reference_wall_thickness_m": (
                PLAN_SCALE_REFERENCE_WALL_THICKNESS
            ),
            "source_wall_depth": self.source_wall_depth,
            "source_center": list(self.source_center),
            "dimensions_m": {
                "width": self.dimensions[0],
                "depth": self.dimensions[1],
                "height": self.dimensions[2],
            },
            "components": {
                name: {
                    "vertices": int(len(mesh.vertices)),
                    "faces": int(len(mesh.faces)),
                    "watertight": bool(mesh.is_watertight),
                }
                for name, mesh in self.components.items()
            },
            "options": asdict(self.options),
            "door_treatments": self.door_treatments,
            "geometry_variations": self.geometry_variations,
            "warnings": list(self.warnings),
            "coordinate_system": "metres, Z-up, floor surface at Z=0",
        }


def _iter_polygons(geometry: BaseGeometry | None) -> Iterable[Polygon]:
    if geometry is None or geometry.is_empty:
        return
    if isinstance(geometry, Polygon):
        yield geometry
    elif isinstance(geometry, MultiPolygon):
        yield from geometry.geoms
    elif isinstance(geometry, GeometryCollection):
        for part in geometry.geoms:
            yield from _iter_polygons(part)


def _drop_tiny_holes(geometry: BaseGeometry) -> BaseGeometry:
    """Remove precision-noise rings while preserving architectural voids."""
    rebuilt: list[Polygon] = []
    for polygon in _iter_polygons(geometry):
        minimum_hole_area = max(1e-10, polygon.area * 1e-12)
        holes = [
            ring.coords
            for ring in polygon.interiors
            if Polygon(ring).area > minimum_hole_area
        ]
        rebuilt.append(Polygon(polygon.exterior.coords, holes))
    return unary_union(rebuilt) if rebuilt else GeometryCollection()


def _clean_polygonal(
    geometry: Any, label: str, warnings: list[str]
) -> BaseGeometry:
    if not isinstance(geometry, BaseGeometry) or geometry.is_empty:
        return GeometryCollection()
    candidate = geometry
    if not candidate.is_valid:
        warnings.append(f"{label}: repaired invalid source geometry")
        candidate = make_valid(candidate)
    polygons = [part for part in _iter_polygons(candidate) if part.area > 1e-9]
    if not polygons:
        warnings.append(f"{label}: no usable polygon geometry")
        return GeometryCollection()
    merged = unary_union(polygons)
    if not merged.is_valid:
        merged = make_valid(merged)
    cleaned = [part for part in _iter_polygons(merged) if part.area > 1e-9]
    return (
        _drop_tiny_holes(unary_union(cleaned))
        if cleaned
        else GeometryCollection()
    )


def _union(geometries: Iterable[BaseGeometry]) -> BaseGeometry:
    usable = [geometry for geometry in geometries if not geometry.is_empty]
    return (
        _drop_tiny_holes(unary_union(usable))
        if usable
        else GeometryCollection()
    )


def _sorted_polygons(geometry: BaseGeometry) -> list[Polygon]:
    """Return individual polygons in a stable, spatial order."""
    return sorted(
        _iter_polygons(geometry),
        key=lambda polygon: (
            round(polygon.centroid.x, 9),
            round(polygon.centroid.y, 9),
            round(polygon.area, 9),
        ),
    )


def _seed_for_plan(plan_id: int | str, user_seed: int) -> int:
    """Mix a user seed with a plan ID without relying on process-random hashes."""
    payload = f"{plan_id}:{user_seed}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _transform(
    geometry: BaseGeometry,
    scale_factor: float,
    center: tuple[float, float],
) -> BaseGeometry:
    if geometry.is_empty:
        return geometry
    scaled = affinity.scale(
        geometry, xfact=scale_factor, yfact=scale_factor, origin=(0.0, 0.0)
    )
    translated = affinity.translate(
        scaled,
        xoff=-center[0] * scale_factor,
        yoff=-center[1] * scale_factor,
    )
    # Dataset unions can retain adjacent coordinates that differ only around
    # machine epsilon. Ear clipping interprets those as zero-width slivers and
    # produces non-manifold seams. A nanometre metric grid removes only that
    # numerical noise.
    return set_precision(translated, grid_size=GEOMETRY_GRID_SIZE)


def _resize_wall_body(
    geometry: BaseGeometry,
    target_thickness: float,
    warnings: list[str],
) -> BaseGeometry:
    """Resize connected wall bands without changing the plan's metre scale."""
    if geometry.is_empty or math.isclose(
        target_thickness,
        PLAN_SCALE_REFERENCE_WALL_THICKNESS,
        abs_tol=1e-12,
    ):
        return geometry
    offset = (
        target_thickness - PLAN_SCALE_REFERENCE_WALL_THICKNESS
    ) / 2.0
    adjusted = geometry.buffer(offset, join_style=2, cap_style=2)
    adjusted = set_precision(adjusted, grid_size=GEOMETRY_GRID_SIZE)
    if adjusted.is_empty:
        raise ValueError(
            "wall_thickness removed all usable wall geometry; "
            "choose a larger value"
        )
    if len(list(_iter_polygons(adjusted))) < len(list(_iter_polygons(geometry))):
        warnings.append(
            "walls: thickness adjustment merged or removed narrow wall parts"
        )
    return _drop_tiny_holes(adjusted)


def _resize_wall_sections(
    geometry: BaseGeometry,
    target_thickness: float,
) -> BaseGeometry:
    """Resize door/window wall-depth while preserving opening span."""
    if geometry.is_empty or math.isclose(
        target_thickness,
        PLAN_SCALE_REFERENCE_WALL_THICKNESS,
        abs_tol=1e-12,
    ):
        return geometry
    ratio = target_thickness / PLAN_SCALE_REFERENCE_WALL_THICKNESS
    resized: list[Polygon] = []
    for polygon in _iter_polygons(geometry):
        rectangle = polygon.minimum_rotated_rectangle
        coordinates = list(rectangle.exterior.coords)[:4]
        if len(coordinates) < 4:
            continue
        edge_lengths = [
            math.hypot(
                coordinates[(index + 1) % 4][0] - coordinates[index][0],
                coordinates[(index + 1) % 4][1] - coordinates[index][1],
            )
            for index in range(4)
        ]
        longest_index = max(range(4), key=edge_lengths.__getitem__)
        start = coordinates[longest_index]
        end = coordinates[(longest_index + 1) % 4]
        angle = math.atan2(end[1] - start[1], end[0] - start[0])
        center = (polygon.centroid.x, polygon.centroid.y)
        aligned = affinity.rotate(
            polygon, -angle, origin=center, use_radians=True
        )
        scaled = affinity.scale(
            aligned, xfact=1.0, yfact=ratio, origin=center
        )
        resized.append(
            affinity.rotate(
                scaled, angle, origin=center, use_radians=True
            )
        )
    if not resized:
        return GeometryCollection()
    return set_precision(_union(resized), grid_size=GEOMETRY_GRID_SIZE)


def _narrow_door_openings(
    geometry: BaseGeometry,
    target_width: float,
    warnings: list[str],
) -> tuple[BaseGeometry, list[dict[str, float]]]:
    """Clip each door along its long axis to a centred metric opening."""
    narrowed: list[BaseGeometry] = []
    widths: list[dict[str, float]] = []
    unchanged_count = 0
    for polygon in _sorted_polygons(geometry):
        rectangle = polygon.minimum_rotated_rectangle
        coordinates = list(rectangle.exterior.coords)[:4]
        if len(coordinates) < 4:
            continue
        edge_lengths = [
            math.hypot(
                coordinates[(index + 1) % 4][0] - coordinates[index][0],
                coordinates[(index + 1) % 4][1] - coordinates[index][1],
            )
            for index in range(4)
        ]
        longest_index = max(range(4), key=edge_lengths.__getitem__)
        original_width = edge_lengths[longest_index]
        effective_width = min(target_width, original_width)
        widths.append(
            {
                "original_width_m": float(original_width),
                "effective_width_m": float(effective_width),
            }
        )
        if target_width >= original_width - GEOMETRY_GRID_SIZE:
            narrowed.append(polygon)
            unchanged_count += 1
            continue

        start = coordinates[longest_index]
        end = coordinates[(longest_index + 1) % 4]
        angle = math.atan2(end[1] - start[1], end[0] - start[0])
        center = (polygon.centroid.x, polygon.centroid.y)
        aligned = affinity.rotate(
            polygon, -angle, origin=center, use_radians=True
        )
        _, min_y, _, max_y = aligned.bounds
        margin = max(original_width, max_y - min_y, target_width, 1.0)
        clip = box(
            center[0] - target_width / 2.0,
            min_y - margin,
            center[0] + target_width / 2.0,
            max_y + margin,
        )
        clipped = aligned.intersection(clip)
        if clipped.is_empty:
            warnings.append("restricted door: width clipping produced no opening")
            continue
        narrowed.append(
            affinity.rotate(
                clipped, angle, origin=center, use_radians=True
            )
        )

    if unchanged_count:
        warnings.append(
            "restricted doors: requested width was not smaller than "
            f"{unchanged_count} selected opening(s); those openings were unchanged"
        )
    if not narrowed:
        return GeometryCollection(), widths
    return (
        set_precision(_union(narrowed), grid_size=GEOMETRY_GRID_SIZE),
        widths,
    )


def _variation_seed(plan_id: int | str, seed: int, label: str) -> int:
    return _seed_for_plan(f"{plan_id}:geometry:{label}", seed)


def _weighted_group_selection(
    candidates: dict[tuple[tuple[int, int], ...], set[str]],
    weights: dict[str, float],
    seed: int,
    conflict_tokens: dict[tuple[tuple[int, int], ...], set[tuple[int, int]]],
) -> dict[tuple[tuple[int, int], ...], str]:
    """Assign at most one effect to each non-overlapping physical feature."""
    active = {name: value for name, value in weights.items() if value > 0}
    if not candidates or not active:
        return {}
    total = sum(active.values())
    target = min(
        len(candidates),
        max(1, math.ceil(len(candidates) * min(total, 100.0) / 100.0)),
    )
    rng = random.Random(seed)
    raw_counts = {
        name: target * value / total for name, value in active.items()
    }
    counts = {name: int(math.floor(value)) for name, value in raw_counts.items()}
    tie_breakers = {name: rng.random() for name in active}
    for name in sorted(
        active,
        key=lambda item: (
            raw_counts[item] - counts[item],
            active[item],
            tie_breakers[item],
        ),
        reverse=True,
    )[: target - sum(counts.values())]:
        counts[name] += 1
    # If there is enough capacity, make each requested effect visible at least
    # once instead of letting rounding silently eliminate a small share.
    if target >= len(active):
        for name in active:
            if counts[name] > 0 or not any(
                name in allowed for allowed in candidates.values()
            ):
                continue
            donor = max(counts, key=counts.get)
            if counts[donor] > 1:
                counts[donor] -= 1
                counts[name] += 1

    required_modes = [name for name, count in counts.items() if count > 0]
    rng.shuffle(required_modes)
    remaining_modes = [
        name for name, count in counts.items() for _ in range(max(0, count - 1))
    ]
    rng.shuffle(remaining_modes)
    mode_order = required_modes + remaining_modes
    remaining = sorted(candidates)
    rng.shuffle(remaining)
    selected: dict[tuple[tuple[int, int], ...], str] = {}
    occupied: set[tuple[int, int]] = set()
    for mode in mode_order:
        for group in tuple(remaining):
            if mode not in candidates[group]:
                continue
            tokens = conflict_tokens[group]
            if tokens.intersection(occupied):
                continue
            selected[group] = mode
            occupied.update(tokens)
            remaining.remove(group)
            break
    return selected


def _corner_pairs(
    rings: list[list[tuple[float, float]]],
    candidates_by_mode: dict[str, set[tuple[int, int]]],
    wall_thickness: float,
) -> dict[tuple[tuple[int, int], ...], set[str]]:
    """Pair the two faces of a physical wall corner."""
    all_candidates = set().union(*candidates_by_mode.values())
    records: dict[
        tuple[int, int],
        tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    ] = {}
    for ring_index, index in all_candidates:
        coordinates = rings[ring_index]
        vertex = coordinates[index]
        previous = coordinates[index - 1]
        following = coordinates[(index + 1) % len(coordinates)]
        before = math.dist(previous, vertex)
        after = math.dist(following, vertex)
        records[(ring_index, index)] = (
            vertex,
            ((previous[0] - vertex[0]) / before, (previous[1] - vertex[1]) / before),
            ((following[0] - vertex[0]) / after, (following[1] - vertex[1]) / after),
        )

    possible: list[tuple[float, tuple[int, int], tuple[int, int]]] = []
    keys = sorted(records)
    for position, first_key in enumerate(keys):
        first = records[first_key]
        for second_key in keys[position + 1 :]:
            second = records[second_key]
            distance = math.dist(first[0], second[0])
            if not wall_thickness * 0.40 <= distance <= max(
                wall_thickness * 3.5,
                PLAN_SCALE_REFERENCE_WALL_THICKNESS * 1.1,
            ):
                continue
            direct = min(
                first[1][0] * second[1][0] + first[1][1] * second[1][1],
                first[2][0] * second[2][0] + first[2][1] * second[2][1],
            )
            swapped = min(
                first[1][0] * second[2][0] + first[1][1] * second[2][1],
                first[2][0] * second[1][0] + first[2][1] * second[1][1],
            )
            if max(direct, swapped) < math.cos(math.radians(5.0)):
                continue
            possible.append(
                (abs(distance - wall_thickness * math.sqrt(2.0)), first_key, second_key)
            )

    paired: set[tuple[int, int]] = set()
    groups: dict[tuple[tuple[int, int], ...], set[str]] = {}
    for _, first_key, second_key in sorted(possible):
        if first_key in paired or second_key in paired:
            continue
        allowed = {
            mode
            for mode, candidates in candidates_by_mode.items()
            if first_key in candidates and second_key in candidates
        }
        if not allowed:
            continue
        group = tuple(sorted((first_key, second_key)))
        groups[group] = allowed
        paired.update(group)
    return groups


def _segment_pairs(
    rings: list[list[tuple[float, float]]],
    candidates_by_mode: dict[str, set[tuple[int, int]]],
    wall_thickness: float,
) -> dict[tuple[tuple[int, int], ...], set[str]]:
    """Pair parallel wall faces so deformations move the whole wall strip."""
    all_candidates = set().union(*candidates_by_mode.values())
    records: dict[tuple[int, int], dict[str, Any]] = {}
    for key in all_candidates:
        ring_index, index = key
        coordinates = rings[ring_index]
        start = coordinates[index]
        end = coordinates[(index + 1) % len(coordinates)]
        length = math.dist(start, end)
        if length <= 1e-8:
            continue
        records[key] = {
            "start": start,
            "end": end,
            "length": length,
            "unit": ((end[0] - start[0]) / length, (end[1] - start[1]) / length),
            "midpoint": ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0),
        }

    possible: list[tuple[float, tuple[int, int], tuple[int, int]]] = []
    keys = sorted(records)
    for position, first_key in enumerate(keys):
        first = records[first_key]
        ux, uy = first["unit"]
        first_interval = sorted(
            point[0] * ux + point[1] * uy
            for point in (first["start"], first["end"])
        )
        for second_key in keys[position + 1 :]:
            second = records[second_key]
            parallel = abs(ux * second["unit"][0] + uy * second["unit"][1])
            if parallel < math.cos(math.radians(3.0)):
                continue
            dx = second["midpoint"][0] - first["midpoint"][0]
            dy = second["midpoint"][1] - first["midpoint"][1]
            perpendicular = abs(dx * -uy + dy * ux)
            if not wall_thickness * 0.40 <= perpendicular <= max(
                wall_thickness * 2.5,
                PLAN_SCALE_REFERENCE_WALL_THICKNESS * 0.75,
            ):
                continue
            second_interval = sorted(
                point[0] * ux + point[1] * uy
                for point in (second["start"], second["end"])
            )
            overlap = max(
                0.0,
                min(first_interval[1], second_interval[1])
                - max(first_interval[0], second_interval[0]),
            )
            overlap_ratio = overlap / min(first["length"], second["length"])
            length_ratio = min(first["length"], second["length"]) / max(
                first["length"], second["length"]
            )
            if overlap_ratio < 0.80 or length_ratio < 0.80:
                continue
            score = (
                abs(perpendicular - wall_thickness)
                + (1.0 - overlap_ratio)
                + (1.0 - length_ratio)
            )
            possible.append((score, first_key, second_key))

    paired: set[tuple[int, int]] = set()
    groups: dict[tuple[tuple[int, int], ...], set[str]] = {}
    for _, first_key, second_key in sorted(possible):
        if first_key in paired or second_key in paired:
            continue
        allowed = {
            mode
            for mode, candidates in candidates_by_mode.items()
            if first_key in candidates and second_key in candidates
        }
        if not allowed:
            continue
        group = tuple(sorted((first_key, second_key)))
        groups[group] = allowed
        paired.update(group)
    return groups


def _corner_is_eligible(
    coordinates: list[tuple[float, float]],
    index: int,
    size: float,
    protected: BaseGeometry,
    wall_thickness: float,
) -> bool:
    previous = coordinates[index - 1]
    vertex = coordinates[index]
    following = coordinates[(index + 1) % len(coordinates)]
    before = math.hypot(previous[0] - vertex[0], previous[1] - vertex[1])
    after = math.hypot(following[0] - vertex[0], following[1] - vertex[1])
    if min(before, after) < max(wall_thickness * 1.5, 0.12):
        return False
    first = ((previous[0] - vertex[0]) / before, (previous[1] - vertex[1]) / before)
    second = ((following[0] - vertex[0]) / after, (following[1] - vertex[1]) / after)
    cosine = max(-1.0, min(1.0, first[0] * second[0] + first[1] * second[1]))
    angle = math.degrees(math.acos(cosine))
    if not 30.0 <= angle <= 150.0:
        return False
    # The requested size is an upper bound. Short corners are still eligible
    # and use the same 40%-of-edge cap applied by `_vary_ring`, rather than
    # disappearing entirely when the GUI size is large (for example 1 m).
    local_cut = min(size, before * 0.4, after * 0.4)
    clearance = local_cut + wall_thickness
    return protected.is_empty or Point(vertex).distance(protected) > clearance


def _segment_is_eligible(
    start: tuple[float, float],
    end: tuple[float, float],
    amplitude: float,
    protected: BaseGeometry,
    wall_thickness: float,
) -> bool:
    segment = LineString((start, end))
    minimum_length = max(0.75, amplitude * 4.0, wall_thickness * 3.0)
    # Openings protect only their local interval. Rejecting a complete wall
    # face because it contains one door/window made 100% coverage affect only
    # a few walls. The deformation is tapered to zero around those intervals
    # in `_vary_ring` instead.
    return segment.length >= minimum_length


def _quadratic_corner(
    incoming: tuple[float, float],
    vertex: tuple[float, float],
    outgoing: tuple[float, float],
    steps: int = 6,
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for index in range(steps + 1):
        t = index / steps
        inverse = 1.0 - t
        points.append(
            (
                inverse * inverse * incoming[0]
                + 2.0 * inverse * t * vertex[0]
                + t * t * outgoing[0],
                inverse * inverse * incoming[1]
                + 2.0 * inverse * t * vertex[1]
                + t * t * outgoing[1],
            )
        )
    return points


def _vary_ring(
    coordinates: list[tuple[float, float]],
    ring_index: int,
    corner_modes: dict[tuple[int, int], tuple[str, float]],
    segment_modes: dict[tuple[int, int], dict[str, Any]],
) -> list[tuple[float, float]]:
    count = len(coordinates)
    incoming: list[tuple[float, float]] = []
    outgoing: list[tuple[float, float]] = []
    for index, vertex in enumerate(coordinates):
        mode = corner_modes.get((ring_index, index))
        if mode is None:
            incoming.append(vertex)
            outgoing.append(vertex)
            continue
        size = mode[1]
        previous = coordinates[index - 1]
        following = coordinates[(index + 1) % count]
        before_length = math.hypot(previous[0] - vertex[0], previous[1] - vertex[1])
        after_length = math.hypot(following[0] - vertex[0], following[1] - vertex[1])
        cut = min(size, before_length * 0.4, after_length * 0.4)
        incoming.append(
            (
                vertex[0] + (previous[0] - vertex[0]) * cut / before_length,
                vertex[1] + (previous[1] - vertex[1]) * cut / before_length,
            )
        )
        outgoing.append(
            (
                vertex[0] + (following[0] - vertex[0]) * cut / after_length,
                vertex[1] + (following[1] - vertex[1]) * cut / after_length,
            )
        )

    varied: list[tuple[float, float]] = []
    for index, vertex in enumerate(coordinates):
        mode = corner_modes.get((ring_index, index))
        if mode is None:
            varied.append(vertex)
        elif mode[0] == "diagonal":
            varied.extend((incoming[index], outgoing[index]))
        else:
            varied.extend(
                _quadratic_corner(incoming[index], vertex, outgoing[index])
            )

        segment_key = (ring_index, index)
        segment_mode = segment_modes.get(segment_key)
        if segment_mode is None:
            continue
        start = outgoing[index]
        end = incoming[(index + 1) % count]
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length <= 1e-8:
            continue
        unit = (dx / length, dy / length)
        start_shoulder = (
            corner_modes[(ring_index, index)][1] * 0.5
            if (ring_index, index) in corner_modes
            else 0.0
        )
        next_key = (ring_index, (index + 1) % count)
        end_shoulder = (
            corner_modes[next_key][1] * 0.5
            if next_key in corner_modes
            else 0.0
        )
        shoulder_total = start_shoulder + end_shoulder
        if shoulder_total >= length * 0.6:
            continue
        if start_shoulder > 0:
            start = (
                start[0] + unit[0] * start_shoulder,
                start[1] + unit[1] * start_shoulder,
            )
            varied.append(start)
        if end_shoulder > 0:
            end = (
                end[0] - unit[0] * end_shoulder,
                end[1] - unit[1] * end_shoulder,
            )
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        divisions = min(48, max(4, math.ceil(length / 0.30)))
        reference_unit = segment_mode["unit"]
        same_direction = (
            dx / length * reference_unit[0]
            + dy / length * reference_unit[1]
        ) >= 0
        normal = segment_mode["normal"]
        for step in range(1, divisions):
            t = step / divisions
            reference_t = t if same_direction else 1.0 - t
            # A squared envelope has zero slope at both ends. Combined corner
            # and wall effects therefore have a straight visual shoulder
            # instead of forming a diagonal wedge where they meet.
            envelope = math.sin(math.pi * reference_t) ** 2
            if segment_mode["mode"] == "curved":
                offset = segment_mode["amplitude"] * envelope
            else:
                samples = segment_mode["samples"]
                sample_position = reference_t * (len(samples) - 1)
                lower = min(len(samples) - 2, int(math.floor(sample_position)))
                fraction = sample_position - lower
                offset = (
                    samples[lower] * (1.0 - fraction)
                    + samples[lower + 1] * fraction
                ) * envelope
            protected = segment_mode["protected"]
            if not protected.is_empty:
                reference_start = segment_mode["reference_start"]
                reference_end = segment_mode["reference_end"]
                protected_point = Point(
                    reference_start[0]
                    + (reference_end[0] - reference_start[0]) * reference_t,
                    reference_start[1]
                    + (reference_end[1] - reference_start[1]) * reference_t,
                )
                protection = min(
                    1.0,
                    max(
                        0.0,
                        protected_point.distance(protected)
                        - segment_mode["protection_radius"],
                    )
                    / segment_mode["protection_fade"],
                )
                # Smoothstep avoids a visible kink where the protected,
                # straight opening interval blends into the varied wall.
                protection = protection * protection * (3.0 - 2.0 * protection)
                offset *= protection
            varied.append(
                (
                    start[0] + dx * t + normal[0] * offset,
                    start[1] + dy * t + normal[1] * offset,
                )
            )
        if end_shoulder > 0:
            varied.append(end)
    return varied


def _vary_wall_geometry(
    geometry: BaseGeometry,
    protected: BaseGeometry,
    options: ExtrusionOptions,
    plan_id: int | str,
    warnings: list[str],
) -> tuple[BaseGeometry, dict[str, Any]]:
    """Apply safe, seeded irregularities to exterior and partition wall faces."""
    stats: dict[str, Any] = {
        "enabled": False,
        "seed": options.geometry_seed,
        "diagonal_corners": {"eligible": 0, "applied": 0},
        "rounded_corners": {"eligible": 0, "applied": 0},
        "curved_walls": {"eligible": 0, "applied": 0},
        "noisy_walls": {"eligible": 0, "applied": 0},
    }
    active = any(
        value > 0
        for value in (
            options.diagonal_corner_percent,
            options.rounded_corner_percent,
            options.curved_wall_percent,
            options.noisy_wall_percent,
        )
    )
    if not active or geometry.is_empty:
        return geometry, stats
    stats["enabled"] = True

    polygons = _sorted_polygons(geometry)
    rings: list[list[tuple[float, float]]] = []
    polygon_ring_indices: list[list[int]] = []
    for polygon in polygons:
        indices: list[int] = []
        for ring in (polygon.exterior, *polygon.interiors):
            coordinates = [(float(x), float(y)) for x, y in list(ring.coords)[:-1]]
            if len(coordinates) < 3:
                continue
            indices.append(len(rings))
            rings.append(coordinates)
        polygon_ring_indices.append(indices)

    diagonal_candidates: set[tuple[int, int]] = set()
    rounded_candidates: set[tuple[int, int]] = set()
    curved_candidates: set[tuple[int, int]] = set()
    noisy_candidates: set[tuple[int, int]] = set()
    for ring_index, coordinates in enumerate(rings):
        for index, vertex in enumerate(coordinates):
            if _corner_is_eligible(
                coordinates,
                index,
                options.diagonal_corner_size,
                protected,
                options.wall_thickness,
            ):
                diagonal_candidates.add((ring_index, index))
            if _corner_is_eligible(
                coordinates,
                index,
                options.rounded_corner_radius,
                protected,
                options.wall_thickness,
            ):
                rounded_candidates.add((ring_index, index))
            following = coordinates[(index + 1) % len(coordinates)]
            if _segment_is_eligible(
                vertex,
                following,
                options.curved_wall_amplitude,
                protected,
                options.wall_thickness,
            ):
                curved_candidates.add((ring_index, index))
            if _segment_is_eligible(
                vertex,
                following,
                options.noisy_wall_amplitude,
                protected,
                options.wall_thickness,
            ):
                noisy_candidates.add((ring_index, index))

    corner_groups = _corner_pairs(
        rings,
        {
            "diagonal": diagonal_candidates,
            "rounded": rounded_candidates,
        },
        options.wall_thickness,
    )
    # Paired corners remain valid when adjacent: edge-length eligibility caps
    # both cuts, and each corner is still assigned only one style.
    corner_conflicts = {
        group: {(position, -1)}
        for position, group in enumerate(sorted(corner_groups))
    }
    selected_corner_groups = _weighted_group_selection(
        corner_groups,
        {
            "diagonal": options.diagonal_corner_percent,
            "rounded": options.rounded_corner_percent,
        },
        _variation_seed(plan_id, options.geometry_seed, "corner-mix"),
        corner_conflicts,
    )
    segment_groups = _segment_pairs(
        rings,
        {
            "curved": curved_candidates,
            "noisy": noisy_candidates,
        },
        options.wall_thickness,
    )
    # Curves/noise have zero displacement and zero slope at their endpoints,
    # so adjacent paired wall spans can safely be modified together.
    segment_conflicts = {
        group: {(position, -1)}
        for position, group in enumerate(sorted(segment_groups))
    }
    selected_segment_groups = _weighted_group_selection(
        segment_groups,
        {
            "curved": options.curved_wall_percent,
            "noisy": options.noisy_wall_percent,
        },
        _variation_seed(plan_id, options.geometry_seed, "wall-mix"),
        segment_conflicts,
    )

    diagonal = {
        key
        for group, mode in selected_corner_groups.items()
        if mode == "diagonal"
        for key in group
    }
    rounded = {
        key
        for group, mode in selected_corner_groups.items()
        if mode == "rounded"
        for key in group
    }
    curved = {
        key
        for group, mode in selected_segment_groups.items()
        if mode == "curved"
        for key in group
    }
    noisy = {
        key
        for group, mode in selected_segment_groups.items()
        if mode == "noisy"
        for key in group
    }
    stats["diagonal_corners"] = {
        "eligible": sum("diagonal" in allowed for allowed in corner_groups.values()),
        "applied": len(diagonal) // 2,
    }
    stats["rounded_corners"] = {
        "eligible": sum("rounded" in allowed for allowed in corner_groups.values()),
        "applied": len(rounded) // 2,
    }
    stats["curved_walls"] = {
        "eligible": sum("curved" in allowed for allowed in segment_groups.values()),
        "applied": len(curved) // 2,
    }
    stats["noisy_walls"] = {
        "eligible": sum("noisy" in allowed for allowed in segment_groups.values()),
        "applied": len(noisy) // 2,
    }
    stats["corner_mix_percent"] = min(
        100.0,
        options.diagonal_corner_percent + options.rounded_corner_percent,
    )
    stats["wall_mix_percent"] = min(
        100.0,
        options.curved_wall_percent + options.noisy_wall_percent,
    )
    requested = (
        ("diagonal corners", options.diagonal_corner_percent, diagonal),
        ("rounded corners", options.rounded_corner_percent, rounded),
        ("curved walls", options.curved_wall_percent, curved),
        ("noisy walls", options.noisy_wall_percent, noisy),
    )
    for label, percent, selected in requested:
        if percent > 0 and not selected:
            warnings.append(
                f"geometry variation: no safe paired {label} away from openings"
            )
    if options.diagonal_corner_percent + options.rounded_corner_percent > 100:
        warnings.append(
            "geometry variation: corner percentages exceeded 100; "
            "normalized as a mutually exclusive mix"
        )
    if options.curved_wall_percent + options.noisy_wall_percent > 100:
        warnings.append(
            "geometry variation: wall percentages exceeded 100; "
            "normalized as a mutually exclusive mix"
        )

    corner_modes = {
        key: ("diagonal", options.diagonal_corner_size) for key in diagonal
    }
    corner_modes.update(
        {key: ("rounded", options.rounded_corner_radius) for key in rounded}
    )
    segment_modes: dict[tuple[int, int], dict[str, Any]] = {}
    for group, mode in selected_segment_groups.items():
        reference_ring, reference_index = group[0]
        reference_coordinates = rings[reference_ring]
        reference_start = reference_coordinates[reference_index]
        reference_end = reference_coordinates[
            (reference_index + 1) % len(reference_coordinates)
        ]
        reference_length = math.dist(reference_start, reference_end)
        unit = (
            (reference_end[0] - reference_start[0]) / reference_length,
            (reference_end[1] - reference_start[1]) / reference_length,
        )
        rng = random.Random(
            _variation_seed(
                plan_id,
                options.geometry_seed,
                f"{mode}-{group}",
            )
        )
        shortest_length = min(
            math.dist(
                rings[ring_index][index],
                rings[ring_index][(index + 1) % len(rings[ring_index])],
            )
            for ring_index, index in group
        )
        spec: dict[str, Any] = {
            "mode": mode,
            "unit": unit,
            "normal": (-unit[1], unit[0]),
            "reference_start": reference_start,
            "reference_end": reference_end,
            "protected": protected,
            "protection_radius": max(options.wall_thickness, 0.05),
        }
        if mode == "curved":
            spec["amplitude"] = rng.choice((-1.0, 1.0)) * min(
                options.curved_wall_amplitude,
                shortest_length * 0.20,
            )
        else:
            amplitude = min(
                options.noisy_wall_amplitude,
                shortest_length * 0.15,
            )
            sample_count = min(48, max(5, math.ceil(shortest_length / 0.30) + 1))
            spec["samples"] = [0.0] + [
                rng.uniform(-amplitude, amplitude)
                for _ in range(sample_count - 2)
            ] + [0.0]
        spec["protection_fade"] = max(
            options.wall_thickness,
            (
                abs(spec["amplitude"])
                if mode == "curved"
                else options.noisy_wall_amplitude
            ),
            0.05,
        )
        for key in group:
            segment_modes[key] = spec
    varied_rings = [
        _vary_ring(
            coordinates,
            ring_index,
            corner_modes,
            segment_modes,
        )
        for ring_index, coordinates in enumerate(rings)
    ]
    rebuilt: list[BaseGeometry] = []
    for indices in polygon_ring_indices:
        if not indices:
            continue
        candidate: BaseGeometry = Polygon(
            varied_rings[indices[0]],
            [varied_rings[index] for index in indices[1:]],
        )
        if not candidate.is_valid:
            candidate = make_valid(candidate)
            warnings.append("geometry variation: repaired self-intersection")
        rebuilt.append(candidate)
    varied_geometry = _union(rebuilt)
    if varied_geometry.is_empty:
        warnings.append("geometry variation: empty result; used original walls")
        return geometry, stats
    if not protected.is_empty:
        # Guarantee exact, flush contacts after the sampled taper. This also
        # handles source wall bands whose measured width differs slightly from
        # the requested thickness.
        protected_zone = protected.buffer(
            max(options.wall_thickness, 0.05),
            join_style=2,
            cap_style=2,
        )
        varied_geometry = _union(
            (
                varied_geometry.difference(protected_zone),
                geometry.intersection(protected_zone),
            )
        )
    return set_precision(varied_geometry, GEOMETRY_GRID_SIZE), stats


def _adapt_slab_to_varied_walls(
    slab: BaseGeometry,
    enclosed: BaseGeometry,
    original_walls: BaseGeometry,
    varied_walls: BaseGeometry,
) -> BaseGeometry:
    """Extend/cut only the slab perimeter affected by wall variations."""
    removed = original_walls.difference(varied_walls)
    external_removed = _union(
        polygon
        for polygon in _iter_polygons(removed)
        if polygon.distance(enclosed.boundary) <= GEOMETRY_GRID_SIZE * 10
    )
    added = varied_walls.difference(original_walls)
    return _union((slab.difference(external_removed), added))


def _extrude(
    geometry: BaseGeometry,
    z_min: float,
    z_max: float,
    name: str,
    warnings: list[str],
) -> trimesh.Trimesh | None:
    height = z_max - z_min
    if geometry.is_empty or height <= 0:
        return None
    meshes: list[trimesh.Trimesh] = []
    for index, polygon in enumerate(_iter_polygons(geometry)):
        if polygon.area <= 1e-10:
            continue
        try:
            mesh = trimesh.creation.extrude_polygon(
                polygon, height=height, engine="earcut"
            )
        except Exception as exc:
            repaired = polygon.buffer(0)
            try:
                mesh = trimesh.creation.extrude_polygon(
                    repaired, height=height, engine="earcut"
                )
                warnings.append(f"{name}[{index}]: triangulated after buffer repair")
            except Exception as repair_exc:
                warnings.append(
                    f"{name}[{index}]: skipped extrusion "
                    f"({type(exc).__name__}; repair: {type(repair_exc).__name__})"
                )
                continue
        if not mesh.is_watertight:
            # Valid polygons can contain a hole touching the exterior at a
            # single point. Ear clipping then creates a shared vertical edge.
            # Apply a microscopic close/open only to the affected polygon.
            normalized = polygon.buffer(
                GEOMETRY_GRID_SIZE, join_style=2
            ).buffer(-GEOMETRY_GRID_SIZE, join_style=2)
            normalized = set_precision(
                normalized, grid_size=GEOMETRY_GRID_SIZE
            )
            try:
                repaired_parts = [
                    trimesh.creation.extrude_polygon(
                        part, height=height, engine="earcut"
                    )
                    for part in _iter_polygons(normalized)
                ]
                repaired_mesh = trimesh.util.concatenate(
                    repaired_parts
                )
                if repaired_mesh.is_watertight:
                    mesh = repaired_mesh
                    warnings.append(
                        f"{name}[{index}]: repaired a non-manifold pinch point"
                    )
                else:
                    warnings.append(
                        f"{name}[{index}]: extrusion is not watertight"
                    )
            except Exception as repair_exc:
                warnings.append(
                    f"{name}[{index}]: topology repair failed "
                    f"({type(repair_exc).__name__})"
                )
        mesh.apply_translation((0.0, 0.0, z_min))
        mesh.remove_unreferenced_vertices()
        meshes.append(mesh)
    if not meshes:
        return None
    combined = trimesh.util.concatenate(meshes)
    combined.visual.face_colors = COMPONENT_COLORS[name]
    combined.metadata["component"] = name
    return combined


def extrude_plan(
    plan: dict[str, Any], options: ExtrusionOptions | None = None
) -> ExtrusionResult:
    """Convert one ResPlan record into a set of metric, Z-up meshes."""

    options = options or ExtrusionOptions()
    options.validate()
    warnings: list[str] = []
    plan_id = plan.get("id", "unknown")
    if isinstance(plan_id, np.integer):
        plan_id = int(plan_id)

    raw_wall_depth = plan.get("wall_depth")
    try:
        source_wall_depth = float(raw_wall_depth)
    except (TypeError, ValueError) as exc:
        raise ValueError("plan has no usable wall_depth") from exc
    if not math.isfinite(source_wall_depth) or source_wall_depth <= 0:
        raise ValueError("plan wall_depth must be finite and greater than zero")
    scale_factor = (
        PLAN_SCALE_REFERENCE_WALL_THICKNESS / source_wall_depth
    )

    cleaned = {
        key: _clean_polygonal(plan.get(key), key, warnings)
        for key in (
            "inner",
            *FLOOR_SPACE_KEYS,
            "wall",
            "door",
            "front_door",
            "window",
            "balcony",
        )
    }
    if cleaned["wall"].is_empty:
        raise ValueError("plan contains no usable wall geometry")

    room_footprint = _union(cleaned[key] for key in FLOOR_SPACE_KEYS)
    if room_footprint.is_empty:
        if cleaned["inner"].is_empty:
            raise ValueError("plan contains no usable room footprint")
        room_footprint = cleaned["inner"]
        warnings.append(
            "floor: no semantic room geometry; used inner footprint fallback"
        )

    enclosed_footprint = _union(
        (
            room_footprint,
            cleaned["wall"],
            cleaned["door"],
            cleaned["front_door"],
            cleaned["window"],
        )
    )

    interior_door_parts = _sorted_polygons(cleaned["door"])
    front_door_parts = _sorted_polygons(cleaned["front_door"])
    boundary_tolerance = max(1e-7, source_wall_depth * 0.05)
    boundary_indices = {
        index
        for index, door in enumerate(interior_door_parts)
        if door.distance(enclosed_footprint.boundary) <= boundary_tolerance
    }

    restricted_candidates = [
        index
        for index in range(len(interior_door_parts))
        if index not in boundary_indices
    ]
    restricted_count = min(
        options.restricted_door_count, len(restricted_candidates)
    )
    if options.restricted_door_count > len(restricted_candidates):
        warnings.append(
            "restricted doors: requested "
            f"{options.restricted_door_count}, available "
            f"{len(restricted_candidates)}; clamped to available doors"
        )
    selection_rng = random.Random(
        _seed_for_plan(plan_id, options.restricted_door_seed)
    )
    restricted_indices = set(
        selection_rng.sample(restricted_candidates, restricted_count)
    )
    closed_indices = boundary_indices if options.close_boundary_doors else set()

    restricted_door_parts = [
        door
        for index, door in enumerate(interior_door_parts)
        if index in restricted_indices
    ]
    normal_door_parts = [
        door
        for index, door in enumerate(interior_door_parts)
        if index not in restricted_indices and index not in closed_indices
    ]
    closed_boundary_parts = (
        [
            door
            for index, door in enumerate(interior_door_parts)
            if index in closed_indices
        ]
        + front_door_parts
        if options.close_boundary_doors
        else []
    )
    if not options.close_boundary_doors:
        normal_door_parts.extend(front_door_parts)

    normal_doors = _union(normal_door_parts)
    restricted_doors = _union(restricted_door_parts)
    closed_boundary_doors = _union(closed_boundary_parts)
    # Door and window polygons occupy gaps in the source wall layer. Rebuild
    # the complete wall band before changing its thickness, then cut the
    # selected openings back out. Shrinking disconnected wall fragments
    # independently would otherwise create visible seams at every opening
    # and at some T-junctions.
    structural_wall_band = _union(
        (
            cleaned["wall"],
            cleaned["door"],
            cleaned["front_door"],
            cleaned["window"],
        )
    )

    floor_footprint = (
        _union((enclosed_footprint, cleaned["balcony"]))
        if options.include_balcony
        else enclosed_footprint
    )
    if floor_footprint.is_empty:
        raise ValueError("plan produced an empty floor footprint")

    min_x, min_y, max_x, max_y = floor_footprint.bounds
    source_center = (
        ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)
        if options.center
        else (0.0, 0.0)
    )

    geometry = {
        "floor": floor_footprint,
        "door_lintels": normal_doors,
        "restricted_door_lintels": restricted_doors,
        "window_sills": cleaned["window"],
        "window_headers": cleaned["window"],
        "ceiling": enclosed_footprint,
    }
    geometry = {
        name: _transform(value, scale_factor, source_center)
        for name, value in geometry.items()
    }
    geometry["door_lintels"] = _resize_wall_sections(
        geometry["door_lintels"], options.wall_thickness
    )
    restricted_full_openings = _resize_wall_sections(
        geometry["restricted_door_lintels"], options.wall_thickness
    )
    restricted_widths: list[dict[str, float]] = []
    if options.restricted_door_mode in {"width", "both"}:
        restricted_openings, restricted_widths = _narrow_door_openings(
            restricted_full_openings,
            options.restricted_door_width,
            warnings,
        )
    else:
        restricted_openings = restricted_full_openings
    geometry["restricted_door_lintels"] = restricted_openings
    geometry["window_sills"] = _resize_wall_sections(
        geometry["window_sills"], options.wall_thickness
    )
    geometry["window_headers"] = geometry["window_sills"]

    metric_wall_openings = _union(
        (
            geometry["door_lintels"],
            restricted_openings,
            geometry["window_sills"]
            if options.window_mode == "opening"
            else GeometryCollection(),
        )
    )
    metric_wall_band = _resize_wall_body(
        _transform(
            structural_wall_band, scale_factor, source_center
        ),
        options.wall_thickness,
        warnings,
    )
    protected_sections = _resize_wall_sections(
        _transform(
            _union(
                (
                    cleaned["door"],
                    cleaned["front_door"],
                    cleaned["window"],
                )
            ),
            scale_factor,
            source_center,
        ),
        options.wall_thickness,
    )
    original_wall_band = metric_wall_band
    metric_wall_band, geometry_variations = _vary_wall_geometry(
        metric_wall_band,
        protected_sections,
        options,
        plan_id,
        warnings,
    )
    if geometry_variations["enabled"]:
        metric_enclosed = geometry["ceiling"]
        geometry["floor"] = _adapt_slab_to_varied_walls(
            geometry["floor"],
            metric_enclosed,
            original_wall_band,
            metric_wall_band,
        )
        geometry["ceiling"] = _adapt_slab_to_varied_walls(
            metric_enclosed,
            metric_enclosed,
            original_wall_band,
            metric_wall_band,
        )
    geometry["walls"] = _drop_tiny_holes(
        metric_wall_band.difference(metric_wall_openings)
    )
    metric_recentering = (0.0, 0.0)
    if options.center and geometry_variations["enabled"]:
        varied_min_x, varied_min_y, varied_max_x, varied_max_y = geometry[
            "floor"
        ].bounds
        metric_recentering = (
            -(varied_min_x + varied_max_x) / 2.0,
            -(varied_min_y + varied_max_y) / 2.0,
        )
        geometry = {
            name: set_precision(
                affinity.translate(
                    value,
                    xoff=metric_recentering[0],
                    yoff=metric_recentering[1],
                ),
                GEOMETRY_GRID_SIZE,
            )
            for name, value in geometry.items()
        }
    geometry_variations["recentering_m"] = {
        "x": metric_recentering[0],
        "y": metric_recentering[1],
    }

    ranges = {
        "floor": (-options.floor_thickness, 0.0),
        "walls": (0.0, options.wall_height),
        "door_lintels": (options.door_height, options.wall_height),
        "restricted_door_lintels": (
            options.restricted_door_height
            if options.restricted_door_mode in {"height", "both"}
            else options.door_height,
            options.wall_height,
        ),
        "window_sills": (0.0, options.window_sill_height),
        "window_headers": (options.window_head_height, options.wall_height),
        "ceiling": (
            options.wall_height,
            options.wall_height + options.ceiling_thickness,
        ),
    }
    components: dict[str, trimesh.Trimesh] = {}
    for name, polygonal in geometry.items():
        if name == "ceiling" and not options.ceiling:
            continue
        if name == "door_lintels" and options.door_mode == "full-height":
            continue
        if (
            name == "restricted_door_lintels"
            and options.door_mode == "full-height"
            and options.restricted_door_mode == "width"
        ):
            continue
        if (
            name in {"window_sills", "window_headers"}
            and options.window_mode == "solid"
        ):
            continue
        mesh = _extrude(polygonal, *ranges[name], name, warnings)
        if mesh is not None:
            components[name] = mesh

    if "floor" not in components or "walls" not in components:
        raise ValueError("plan did not produce the required floor and wall meshes")
    all_mesh = trimesh.util.concatenate(tuple(components.values()))
    if not np.isfinite(all_mesh.vertices).all():
        raise ValueError("extrusion produced non-finite vertices")

    metric_min_x, metric_min_y, metric_max_x, metric_max_y = geometry[
        "floor"
    ].bounds
    width = metric_max_x - metric_min_x
    depth = metric_max_y - metric_min_y
    height = options.wall_height + (
        options.ceiling_thickness if options.ceiling else 0.0
    )
    restricted_metric = geometry["restricted_door_lintels"]
    closed_metric = _transform(
        closed_boundary_doors, scale_factor, source_center
    )
    if metric_recentering != (0.0, 0.0):
        closed_metric = affinity.translate(
            closed_metric,
            xoff=metric_recentering[0],
            yoff=metric_recentering[1],
        )
    door_treatments = {
        "normal_count": len(normal_door_parts),
        "restricted_count": len(restricted_door_parts),
        "restricted_mode": options.restricted_door_mode,
        "restricted_candidate_count": len(restricted_candidates),
        "restricted_height_m": (
            options.restricted_door_height
            if restricted_door_parts
            and options.restricted_door_mode in {"height", "both"}
            else None
        ),
        "restricted_width_m": (
            options.restricted_door_width
            if restricted_door_parts
            and options.restricted_door_mode in {"width", "both"}
            else None
        ),
        "restricted_widths_m": restricted_widths,
        "restricted_seed": options.restricted_door_seed,
        "restricted_centers_m": [
            {
                "x": float(polygon.centroid.x),
                "y": float(polygon.centroid.y),
            }
            for polygon in _sorted_polygons(restricted_metric)
        ],
        "closed_boundary_count": len(closed_boundary_parts),
        "closed_boundary_centers_m": [
            {
                "x": float(polygon.centroid.x),
                "y": float(polygon.centroid.y),
            }
            for polygon in _sorted_polygons(closed_metric)
        ],
    }
    return ExtrusionResult(
        plan_id=plan_id,
        components=components,
        scale_factor=scale_factor,
        source_wall_depth=source_wall_depth,
        source_center=source_center,
        dimensions=(width, depth, height),
        options=options,
        warnings=warnings,
        door_treatments=door_treatments,
        geometry_variations=geometry_variations,
    )
