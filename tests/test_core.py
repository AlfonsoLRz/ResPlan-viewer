from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box
from shapely.ops import unary_union
import trimesh

import resplan_extruder.core as core
from resplan_extruder import ExtrusionOptions, export_bytes, extrude_plan


def synthetic_plan(
    *,
    include_openings: bool = True,
    include_balcony: bool = True,
) -> dict:
    outer = box(0, 0, 100, 80)
    inner = box(5, 5, 95, 75)
    wall_band = outer.difference(inner)
    partition = box(48, 5, 52, 75)
    door = box(48, 35, 52, 55) if include_openings else GeometryCollection()
    window = box(25, 75, 45, 80) if include_openings else GeometryCollection()
    walls = unary_union((wall_band, partition)).difference(
        unary_union((door, window))
    )
    return {
        "id": 42,
        "wall_depth": 4.0,
        "inner": inner,
        "wall": walls,
        "door": door,
        "front_door": GeometryCollection(),
        "window": window,
        "balcony": box(100, 20, 120, 60)
        if include_balcony
        else GeometryCollection(),
        "net_area": 100.0,
    }


def robotics_plan() -> dict:
    """Synthetic plan with an entrance, a balcony door, and two room doors."""
    plan = synthetic_plan()
    front_door = box(45, 0, 55, 5)
    balcony_door = box(95, 35, 100, 45)
    second_room_door = box(48, 12, 52, 25)
    plan["wall"] = plan["wall"].difference(
        unary_union((front_door, balcony_door, second_room_door))
    )
    plan["front_door"] = front_door
    plan["door"] = unary_union(
        (plan["door"], balcony_door, second_room_door)
    )
    return plan


def z_bounds(mesh: trimesh.Trimesh) -> tuple[float, float]:
    return float(mesh.bounds[0, 2]), float(mesh.bounds[1, 2])


def test_default_extrusion_scale_components_and_heights() -> None:
    result = extrude_plan(synthetic_plan())
    assert result.scale_factor == pytest.approx(0.05)
    assert set(result.components) == {
        "floor",
        "walls",
        "door_lintels",
        "window_sills",
        "window_headers",
    }
    assert z_bounds(result.components["floor"]) == pytest.approx((-0.2, 0.0))
    assert z_bounds(result.components["walls"]) == pytest.approx((0.0, 2.7))
    assert z_bounds(result.components["door_lintels"]) == pytest.approx((2.1, 2.7))
    assert z_bounds(result.components["window_sills"]) == pytest.approx((0.0, 0.9))
    assert z_bounds(result.components["window_headers"]) == pytest.approx(
        (2.1, 2.7)
    )
    assert np.allclose(result.mesh.bounds.mean(axis=0)[:2], (0.0, 0.0))


def test_wall_thickness_does_not_change_plan_scale_or_wall_height() -> None:
    thin = extrude_plan(
        synthetic_plan(), ExtrusionOptions(wall_thickness=0.10)
    )
    default = extrude_plan(
        synthetic_plan(), ExtrusionOptions(wall_thickness=0.20)
    )
    thick = extrude_plan(
        synthetic_plan(), ExtrusionOptions(wall_thickness=0.30)
    )
    assert thin.scale_factor == pytest.approx(default.scale_factor)
    assert thick.scale_factor == pytest.approx(default.scale_factor)
    assert thin.dimensions == pytest.approx(default.dimensions)
    assert thick.dimensions == pytest.approx(default.dimensions)
    assert z_bounds(thin.components["walls"]) == pytest.approx((0.0, 2.7))
    assert z_bounds(thick.components["walls"]) == pytest.approx((0.0, 2.7))
    assert (
        thin.components["walls"].volume
        < default.components["walls"].volume
        < thick.components["walls"].volume
    )


def test_thin_walls_remain_joined_to_door_and_window_sections() -> None:
    result = extrude_plan(
        synthetic_plan(), ExtrusionOptions(wall_thickness=0.05)
    )

    def xy_vertices(component: str) -> set[tuple[float, float]]:
        return {
            (round(float(vertex[0]), 8), round(float(vertex[1]), 8))
            for vertex in result.components[component].vertices
        }

    wall_vertices = xy_vertices("walls")
    # All four corners are shared: there is no slit between the resized wall
    # body and the door lintel/window sill that occupies its opening.
    assert len(wall_vertices & xy_vertices("door_lintels")) >= 4
    assert len(wall_vertices & xy_vertices("window_sills")) >= 4


def test_semantic_rooms_override_an_unrelated_inner_mask() -> None:
    plan = synthetic_plan(include_balcony=False)
    plan["living"] = plan["inner"]
    plan["inner"] = box(-500, 30, 500, 50)

    result = extrude_plan(plan)

    # The structural footprint is 100 x 80 source units at 0.05 m/unit.
    # The unrelated 1,000-unit-wide inner mask must not grow the slab.
    assert result.dimensions == pytest.approx((5.0, 4.0, 2.7))
    assert result.components["floor"].bounds[:, 0] == pytest.approx(
        (-2.5, 2.5)
    )


def test_seeded_geometry_variations_are_reproducible_and_finite() -> None:
    options = ExtrusionOptions(
        diagonal_corner_percent=30,
        rounded_corner_percent=30,
        curved_wall_percent=30,
        noisy_wall_percent=30,
        geometry_seed=19,
    )
    first = extrude_plan(synthetic_plan(), options)
    second = extrude_plan(synthetic_plan(), options)
    another = extrude_plan(
        synthetic_plan(),
        ExtrusionOptions(
            diagonal_corner_percent=30,
            rounded_corner_percent=30,
            curved_wall_percent=30,
            noisy_wall_percent=30,
            geometry_seed=20,
        ),
    )

    assert np.array_equal(
        first.components["walls"].vertices,
        second.components["walls"].vertices,
    )
    assert not np.array_equal(
        first.components["walls"].vertices,
        another.components["walls"].vertices,
    )
    assert first.geometry_variations["enabled"] is True
    assert all(
        first.geometry_variations[name]["applied"] > 0
        for name in (
            "diagonal_corners",
            "rounded_corners",
            "curved_walls",
            "noisy_walls",
        )
    )
    assert all(
        mesh.is_watertight and np.isfinite(mesh.vertices).all()
        for mesh in first.components.values()
    )
    assert np.allclose(first.mesh.bounds.mean(axis=0)[:2], (0.0, 0.0))


def test_geometry_variations_preserve_opening_contacts() -> None:
    result = extrude_plan(
        synthetic_plan(),
        ExtrusionOptions(
            wall_thickness=0.05,
            diagonal_corner_percent=100,
            rounded_corner_percent=100,
            curved_wall_percent=100,
            noisy_wall_percent=100,
            geometry_seed=8,
        ),
    )

    def xy_vertices(component: str) -> set[tuple[float, float]]:
        return {
            (round(float(vertex[0]), 8), round(float(vertex[1]), 8))
            for vertex in result.components[component].vertices
        }

    walls = xy_vertices("walls")
    assert len(walls & xy_vertices("door_lintels")) >= 4
    assert len(walls & xy_vertices("window_sills")) >= 4
    assert all(
        result.geometry_variations[name]["applied"] > 0
        for name in (
            "diagonal_corners",
            "rounded_corners",
            "curved_walls",
            "noisy_walls",
        )
    )
    assert result.geometry_variations["corner_mix_percent"] == 100
    assert result.geometry_variations["wall_mix_percent"] == 100
    assert any("mutually exclusive mix" in warning for warning in result.warnings)


def test_curves_move_paired_wall_faces_and_preserve_thickness() -> None:
    varied, stats = core._vary_wall_geometry(
        box(0, 0, 8, 0.2),
        GeometryCollection(),
        ExtrusionOptions(curved_wall_percent=100, curved_wall_amplitude=0.2),
        plan_id=7,
        warnings=[],
    )

    polygon = next(core._iter_polygons(varied))
    rows: dict[float, list[float]] = {}
    for x, y in list(polygon.exterior.coords)[:-1]:
        rows.setdefault(round(float(x), 6), []).append(float(y))
    paired_rows = [values for values in rows.values() if len(values) == 2]

    assert stats["curved_walls"]["applied"] == 1
    assert len(paired_rows) >= 4
    assert all(max(values) - min(values) == pytest.approx(0.2) for values in paired_rows)


def test_overfull_effect_mix_never_stacks_and_seed_can_change_style() -> None:
    feature = ((0, 0), (1, 0))
    candidates = {feature: {"diagonal", "rounded"}}
    conflicts = {feature: {(0, 0), (1, 0)}}
    selections = {
        next(
            iter(
                core._weighted_group_selection(
                    candidates,
                    {"diagonal": 100, "rounded": 100},
                    seed,
                    conflicts,
                ).values()
            )
        )
        for seed in range(12)
    }

    assert selections == {"diagonal", "rounded"}


@pytest.mark.parametrize(
    ("options", "stat_name"),
    (
        (
            ExtrusionOptions(
                diagonal_corner_percent=100,
                diagonal_corner_size=1.0,
            ),
            "diagonal_corners",
        ),
        (ExtrusionOptions(curved_wall_percent=100), "curved_walls"),
        (ExtrusionOptions(noisy_wall_percent=100), "noisy_walls"),
    ),
)
def test_single_effect_at_100_percent_changes_every_eligible_feature(
    options: ExtrusionOptions,
    stat_name: str,
) -> None:
    result = extrude_plan(synthetic_plan(), options)
    stats = result.geometry_variations[stat_name]

    assert stats["eligible"] > 2
    assert stats["applied"] == stats["eligible"]


def test_ceiling_and_balcony_behavior() -> None:
    with_balcony = extrude_plan(
        synthetic_plan(),
        ExtrusionOptions(ceiling=True, include_balcony=True),
    )
    without_balcony = extrude_plan(
        synthetic_plan(),
        ExtrusionOptions(ceiling=True, include_balcony=False),
    )
    assert z_bounds(with_balcony.components["ceiling"]) == pytest.approx((2.7, 2.85))
    assert with_balcony.components["floor"].volume > without_balcony.components[
        "floor"
    ].volume
    assert with_balcony.components["ceiling"].volume == pytest.approx(
        without_balcony.components["ceiling"].volume
    )


def test_empty_optional_openings_are_supported() -> None:
    result = extrude_plan(synthetic_plan(include_openings=False))
    assert "door_lintels" not in result.components
    assert "window_sills" not in result.components
    assert "window_headers" not in result.components


def test_full_height_door_mode_removes_lintels() -> None:
    result = extrude_plan(
        synthetic_plan(), ExtrusionOptions(door_mode="full-height")
    )
    assert "door_lintels" not in result.components
    assert "window_sills" in result.components


def test_solid_window_mode_fills_window_gap() -> None:
    opening = extrude_plan(
        synthetic_plan(), ExtrusionOptions(window_mode="opening")
    )
    solid = extrude_plan(
        synthetic_plan(), ExtrusionOptions(window_mode="solid")
    )
    assert "window_sills" not in solid.components
    assert "window_headers" not in solid.components
    assert solid.components["walls"].volume > opening.components["walls"].volume
    assert "door_lintels" in solid.components


def test_boundary_door_closure_fills_entrance_and_balcony_openings() -> None:
    open_result = extrude_plan(robotics_plan())
    closed_result = extrude_plan(
        robotics_plan(), ExtrusionOptions(close_boundary_doors=True)
    )
    assert closed_result.components["walls"].volume > open_result.components[
        "walls"
    ].volume
    assert closed_result.door_treatments["closed_boundary_count"] == 2
    assert len(
        closed_result.door_treatments["closed_boundary_centers_m"]
    ) == 2
    assert closed_result.door_treatments["normal_count"] == 2


def test_configurable_restricted_doors_use_alternate_height() -> None:
    result = extrude_plan(
        robotics_plan(),
        ExtrusionOptions(
            close_boundary_doors=True,
            restricted_door_count=1,
            restricted_door_height=0.65,
        ),
    )
    assert "restricted_door_lintels" in result.components
    assert z_bounds(result.components["restricted_door_lintels"]) == pytest.approx(
        (0.65, 2.7)
    )
    assert result.door_treatments["restricted_count"] == 1
    assert result.door_treatments["restricted_candidate_count"] == 2
    assert result.door_treatments["restricted_seed"] == 0
    assert len(result.door_treatments["restricted_centers_m"]) == 1
    assert result.door_treatments["closed_boundary_count"] == 2


def test_restricted_door_count_clamps_to_available_internal_doors() -> None:
    result = extrude_plan(
        robotics_plan(),
        ExtrusionOptions(
            restricted_door_count=20,
            restricted_door_height=0.5,
        ),
    )
    assert result.door_treatments["restricted_count"] == 2
    assert any("clamped to available doors" in warning for warning in result.warnings)


def test_restricted_door_seed_is_reproducible_and_changes_selection() -> None:
    plan = robotics_plan()
    selections = []
    for seed in range(8):
        first = extrude_plan(
            plan,
            ExtrusionOptions(
                restricted_door_count=1,
                restricted_door_height=0.5,
                restricted_door_seed=seed,
            ),
        )
        second = extrude_plan(
            plan,
            ExtrusionOptions(
                restricted_door_count=1,
                restricted_door_height=0.5,
                restricted_door_seed=seed,
            ),
        )
        first_centers = first.door_treatments["restricted_centers_m"]
        assert first_centers == second.door_treatments["restricted_centers_m"]
        selections.append(tuple(first_centers[0].values()))
    assert len(set(selections)) > 1


def test_multipolygon_holes_and_invalid_geometry_are_repaired() -> None:
    plan = synthetic_plan()
    plan["inner"] = MultiPolygon((box(5, 5, 45, 75), box(55, 5, 95, 75)))
    plan["door"] = Polygon([(48, 35), (52, 55), (48, 55), (52, 35), (48, 35)])
    result = extrude_plan(plan)
    assert result.mesh.faces.shape[0] > 0
    assert any("repaired invalid" in warning for warning in result.warnings)


def test_microscopic_holes_are_removed_from_slabs() -> None:
    plan = synthetic_plan(include_balcony=False)
    tiny_hole = box(20, 20, 20.00000001, 20.00000001)
    plan["inner"] = plan["inner"].difference(tiny_hole)
    result = extrude_plan(plan)
    assert result.components["floor"].is_watertight


def test_option_validation() -> None:
    with pytest.raises(ValueError, match="door_height"):
        ExtrusionOptions(door_height=2.8).validate()
    with pytest.raises(ValueError, match="window heights"):
        ExtrusionOptions(window_sill_height=2.2, window_head_height=2.1).validate()
    with pytest.raises(ValueError, match="door_mode"):
        ExtrusionOptions(door_mode="missing").validate()
    with pytest.raises(ValueError, match="window_mode"):
        ExtrusionOptions(window_mode="missing").validate()
    with pytest.raises(ValueError, match="restricted_door_count"):
        ExtrusionOptions(restricted_door_count=-1).validate()
    with pytest.raises(ValueError, match="restricted_door_seed"):
        ExtrusionOptions(restricted_door_seed=-1).validate()
    with pytest.raises(ValueError, match="geometry_seed"):
        ExtrusionOptions(geometry_seed=-1).validate()
    with pytest.raises(ValueError, match="diagonal_corner_percent"):
        ExtrusionOptions(diagonal_corner_percent=101).validate()
    with pytest.raises(ValueError, match="noisy_wall_amplitude"):
        ExtrusionOptions(
            noisy_wall_percent=10,
            noisy_wall_amplitude=0,
        ).validate()
    with pytest.raises(ValueError, match="restricted_door_height"):
        ExtrusionOptions(
            restricted_door_count=1,
            restricted_door_height=2.7,
        ).validate()
    ExtrusionOptions(
        wall_height=1.5,
        door_mode="full-height",
        window_mode="solid",
    ).validate()


def test_export_round_trip() -> None:
    result = extrude_plan(synthetic_plan(), ExtrusionOptions(ceiling=True))
    obj = export_bytes(result, "obj")
    glb = export_bytes(result, "glb")
    assert obj.startswith(b"# ResPlan")
    assert len(glb) > 100
    loaded_obj = trimesh.load_mesh(
        trimesh.util.wrap_as_stream(obj), file_type="obj"
    )
    loaded_glb = trimesh.load(
        trimesh.util.wrap_as_stream(glb), file_type="glb"
    )
    assert len(loaded_obj.vertices) > 0
    assert len(loaded_glb.geometry) > 0
