"""Streamlit source-plan and extrusion viewer."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import secrets
from typing import Any, Iterable

import numpy as np
import plotly.graph_objects as go
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
import streamlit as st

from resplan_extruder.core import (
    COMPONENT_COLORS,
    ExtrusionOptions,
    extrude_plan,
)
from resplan_extruder.exporters import export_bytes
from resplan_extruder.loader import load_dataset


PLAN_COLORS = {
    "living": "#d9d9d9",
    "bedroom": "#66c2a5",
    "bathroom": "#fc8d62",
    "kitchen": "#8da0cb",
    "storage": "#ff8c69",
    "stair": "#9e9ac8",
    "balcony": "#b3b3b3",
    "wall": "#ffd92f",
    "door": "#e78ac3",
    "window": "#a6d854",
    "front_door": "#a63603",
}

MAX_RESTRICTED_DOOR_SEED = 2_147_483_647


def _pick_restricted_door_seed() -> None:
    current = int(st.session_state.get("restricted_door_seed", 0))
    candidate = secrets.randbelow(MAX_RESTRICTED_DOOR_SEED + 1)
    if candidate == current:
        candidate = (candidate + 1) % (MAX_RESTRICTED_DOOR_SEED + 1)
    st.session_state["restricted_door_seed"] = candidate


def _pick_geometry_seed() -> None:
    current = int(st.session_state.get("geometry_seed", 0))
    candidate = secrets.randbelow(MAX_RESTRICTED_DOOR_SEED + 1)
    if candidate == current:
        candidate = (candidate + 1) % (MAX_RESTRICTED_DOOR_SEED + 1)
    st.session_state["geometry_seed"] = candidate


@st.cache_resource(show_spinner="Loading the ResPlan dataset…")
def _load(path: str) -> list[dict[str, Any]]:
    return load_dataset(path)


def _polygons(geometry: BaseGeometry | None) -> Iterable[Polygon]:
    if geometry is None or geometry.is_empty:
        return
    if isinstance(geometry, Polygon):
        yield geometry
    elif isinstance(geometry, MultiPolygon):
        yield from geometry.geoms
    elif isinstance(geometry, GeometryCollection):
        for part in geometry.geoms:
            yield from _polygons(part)


def plan_figure(plan: dict[str, Any]) -> go.Figure:
    figure = go.Figure()
    for category, color in PLAN_COLORS.items():
        shown = False
        geometry = plan.get(category)
        if not isinstance(geometry, BaseGeometry):
            continue
        for polygon in _polygons(geometry):
            x, y = polygon.exterior.xy
            figure.add_trace(
                go.Scatter(
                    x=list(x),
                    y=list(y),
                    mode="lines",
                    fill="toself",
                    fillcolor=color,
                    line={"color": "#333333", "width": 0.7},
                    name=category.replace("_", " "),
                    legendgroup=category,
                    showlegend=not shown,
                    hoverinfo="name",
                )
            )
            shown = True
    figure.update_layout(
        margin={"l": 0, "r": 0, "t": 35, "b": 0},
        title="Source geometry",
        xaxis={"visible": False, "scaleanchor": "y"},
        yaxis={"visible": False, "autorange": "reversed"},
        legend={"orientation": "h"},
        height=600,
    )
    return figure


def mesh_figure(result: Any, show_ceiling: bool) -> go.Figure:
    figure = go.Figure()
    for name, mesh in result.components.items():
        if name == "ceiling" and not show_ceiling:
            continue
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)
        rgba = COMPONENT_COLORS[name]
        color = f"rgb({rgba[0]},{rgba[1]},{rgba[2]})"
        figure.add_trace(
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                name=name.replace("_", " "),
                color=color,
                opacity=rgba[3] / 255,
                flatshading=True,
                hoverinfo="name",
                lighting={
                    "ambient": 0.55,
                    "diffuse": 0.75,
                    "roughness": 0.8,
                    "specular": 0.1,
                },
            )
        )
    figure.update_layout(
        title="Extruded model",
        margin={"l": 0, "r": 0, "t": 35, "b": 0},
        height=600,
        scene={
            "aspectmode": "data",
            "xaxis_title": "X (m)",
            "yaxis_title": "Y (m)",
            "zaxis_title": "Z (m)",
            "camera": {"eye": {"x": 1.5, "y": -1.5, "z": 1.4}},
        },
        legend={"orientation": "h"},
    )
    return figure


def main() -> None:
    st.set_page_config(page_title="ResPlan Extruder", layout="wide")
    st.title("ResPlan Floor-Plan Extruder")
    st.caption("Inspect source vectors, tune metric dimensions, and export OBJ or GLB.")

    default_data = str(Path.cwd() / "ResPlan.pkl")
    with st.sidebar.expander("Dataset", expanded=False):
        data_path = st.text_input("Dataset path", value=default_data)
    try:
        plans = _load(data_path)
    except Exception as exc:
        st.error(f"Could not load dataset: {exc}")
        st.stop()

    index = {str(plan.get("id")): plan for plan in plans}
    plan_ids = list(index)
    selected_id = st.sidebar.selectbox("Plan ID", options=plan_ids)
    plan = index[selected_id]

    with st.sidebar.expander("Structure", expanded=True):
        wall_thickness = st.number_input(
            "Wall thickness (m)", min_value=0.01, value=0.20, step=0.01
        )
        wall_height = st.number_input(
            "Wall height (m)", min_value=0.50, value=2.70, step=0.05
        )
        floor_thickness = st.number_input(
            "Floor thickness (m)", min_value=0.01, value=0.20, step=0.01
        )
        include_balcony = st.checkbox("Include balcony floor", value=True)
        include_ceiling = st.checkbox("Include ceiling", value=False)
        ceiling_thickness = st.number_input(
            "Ceiling thickness (m)",
            min_value=0.01,
            value=0.15,
            step=0.01,
            disabled=not include_ceiling,
        )

    with st.sidebar.expander("Doors and windows", expanded=False):
        door_mode_label = st.selectbox(
            "Door treatment",
            options=("Lintel above door", "Open to wall height"),
            help=(
                "Retain wall above the configured door height, or leave the "
                "complete door footprint open."
            ),
        )
        door_mode = (
            "lintel"
            if door_mode_label == "Lintel above door"
            else "full-height"
        )
        door_height = st.number_input(
            "Door height (m)",
            min_value=0.10,
            value=2.10,
            step=0.05,
            disabled=door_mode == "full-height",
            help="Used only when door treatment retains a lintel.",
        )
        window_mode_label = st.selectbox(
            "Window treatment",
            options=("Opening", "Solid wall"),
            help=(
                "Create sill/header window openings, or fill every supplied "
                "window footprint with full-height wall."
            ),
        )
        window_mode = "opening" if window_mode_label == "Opening" else "solid"
        window_sill = st.number_input(
            "Window sill (m)",
            min_value=0.0,
            value=0.90,
            step=0.05,
            disabled=window_mode == "solid",
        )
        window_head = st.number_input(
            "Window head (m)",
            min_value=0.10,
            value=2.10,
            step=0.05,
            disabled=window_mode == "solid",
        )

    with st.sidebar.expander("Robotics access", expanded=False):
        close_boundary_doors = st.checkbox(
            "Close exterior and balcony doors",
            value=False,
            help=(
                "Fill front-door and boundary-door footprints with "
                "full-height wall so a robot cannot leave the floor."
            ),
        )
        restricted_door_count = st.number_input(
            "Restricted interior doors",
            min_value=0,
            value=0,
            step=1,
        )
        restricted_door_height = st.number_input(
            "Restricted clearance (m)",
            min_value=0.0,
            value=1.00,
            step=0.05,
            disabled=restricted_door_count == 0,
        )
        restricted_door_seed = st.number_input(
            "Restricted-door seed",
            min_value=0,
            max_value=MAX_RESTRICTED_DOOR_SEED,
            value=0,
            step=1,
            disabled=restricted_door_count == 0,
            key="restricted_door_seed",
        )
        st.button(
            "Randomize restricted doors",
            on_click=_pick_restricted_door_seed,
            disabled=restricted_door_count == 0,
            use_container_width=True,
        )

    with st.sidebar.expander("Geometry Lab", expanded=False):
        st.caption(
            "Seeded, batch-reproducible changes. Door and window areas are "
            "protected from deformation."
        )
        diagonal_corner_percent = st.slider(
            "Diagonal corners (%)", 0, 100, 0, 5
        )
        diagonal_corner_size = st.number_input(
            "Diagonal cut size (m)",
            min_value=0.01,
            value=0.30,
            step=0.05,
            disabled=diagonal_corner_percent == 0,
        )
        rounded_corner_percent = st.slider(
            "Rounded corners (%)", 0, 100, 0, 5
        )
        corner_mix_total = diagonal_corner_percent + rounded_corner_percent
        st.caption(
            f"Corner coverage: {min(corner_mix_total, 100)}%. "
            + (
                "Shares above 100% are normalized; a corner receives only one style."
                if corner_mix_total > 100
                else "Each selected corner receives only one style."
            )
        )
        rounded_corner_radius = st.number_input(
            "Corner radius (m)",
            min_value=0.01,
            value=0.30,
            step=0.05,
            disabled=rounded_corner_percent == 0,
        )
        curved_wall_percent = st.slider(
            "Curved wall faces (%)", 0, 100, 0, 5
        )
        curved_wall_amplitude = st.number_input(
            "Curve depth (m)",
            min_value=0.01,
            value=0.15,
            step=0.05,
            disabled=curved_wall_percent == 0,
        )
        noisy_wall_percent = st.slider(
            "Noisy wall faces (%)", 0, 100, 0, 5
        )
        wall_mix_total = curved_wall_percent + noisy_wall_percent
        st.caption(
            f"Wall coverage: {min(wall_mix_total, 100)}%. "
            + (
                "Shares above 100% are normalized; a wall receives only one style."
                if wall_mix_total > 100
                else "Paired wall faces move together and receive only one style."
            )
        )
        noisy_wall_amplitude = st.number_input(
            "Noise amplitude (m)",
            min_value=0.01,
            value=0.08,
            step=0.01,
            disabled=noisy_wall_percent == 0,
        )
        geometry_seed = st.number_input(
            "Geometry seed",
            min_value=0,
            max_value=MAX_RESTRICTED_DOOR_SEED,
            value=0,
            step=1,
            key="geometry_seed",
        )
        geometry_active = any(
            value > 0
            for value in (
                diagonal_corner_percent,
                rounded_corner_percent,
                curved_wall_percent,
                noisy_wall_percent,
            )
        )
        st.button(
            "Randomize geometry",
            on_click=_pick_geometry_seed,
            disabled=not geometry_active,
            use_container_width=True,
            help="Choose a new reproducible layout for the active effects.",
        )

    with st.sidebar.expander("Preview", expanded=False):
        show_ceiling = st.checkbox(
            "Show ceiling in 3D", value=False, disabled=not include_ceiling
        )

    options = ExtrusionOptions(
        wall_thickness=wall_thickness,
        wall_height=wall_height,
        floor_thickness=floor_thickness,
        ceiling=include_ceiling,
        ceiling_thickness=ceiling_thickness,
        door_height=door_height,
        door_mode=door_mode,
        close_boundary_doors=close_boundary_doors,
        restricted_door_count=int(restricted_door_count),
        restricted_door_height=restricted_door_height,
        restricted_door_seed=int(restricted_door_seed),
        window_sill_height=window_sill,
        window_head_height=window_head,
        window_mode=window_mode,
        include_balcony=include_balcony,
        diagonal_corner_percent=float(diagonal_corner_percent),
        diagonal_corner_size=diagonal_corner_size,
        rounded_corner_percent=float(rounded_corner_percent),
        rounded_corner_radius=rounded_corner_radius,
        curved_wall_percent=float(curved_wall_percent),
        curved_wall_amplitude=curved_wall_amplitude,
        noisy_wall_percent=float(noisy_wall_percent),
        noisy_wall_amplitude=noisy_wall_amplitude,
        geometry_seed=int(geometry_seed),
    )
    try:
        result = extrude_plan(plan, options)
    except Exception as exc:
        st.error(f"Could not extrude plan {selected_id}: {exc}")
        st.stop()
    if (
        restricted_door_count > 0
        and result.door_treatments["restricted_count"]
        == result.door_treatments["restricted_candidate_count"]
    ):
        st.info(
            "All eligible interior doors are restricted for this plan; "
            "changing the seed will not change the selection."
        )
    if geometry_active:
        variation_labels = {
            "diagonal_corners": ("diagonal", diagonal_corner_percent),
            "rounded_corners": ("rounded", rounded_corner_percent),
            "curved_walls": ("curved", curved_wall_percent),
            "noisy_walls": ("noisy", noisy_wall_percent),
        }
        st.caption(
            "Applied geometry: "
            + ", ".join(
                f"{result.geometry_variations[key]['applied']} {label}"
                for key, (label, _) in variation_labels.items()
            )
        )
        unavailable = [
            label
            for key, (label, requested_percent) in variation_labels.items()
            if requested_percent > 0
            and result.geometry_variations[key]["applied"] == 0
        ]
        if unavailable:
            st.info(
                "No eligible paired surface was available for: "
                + ", ".join(unavailable)
                + ". Other requested effects were still applied safely."
            )

    left, right = st.columns(2)
    with left:
        st.plotly_chart(plan_figure(plan), use_container_width=True)
    with right:
        st.plotly_chart(
            mesh_figure(result, show_ceiling), use_container_width=True
        )

    obj_data = export_bytes(result, "obj")
    glb_data = export_bytes(result, "glb")
    download_obj, download_glb, details = st.columns([1, 1, 2])
    with download_obj:
        st.download_button(
            "Download OBJ",
            data=obj_data,
            file_name=f"resplan_{selected_id}.obj",
            mime="text/plain",
            use_container_width=True,
        )
    with download_glb:
        st.download_button(
            "Download GLB",
            data=glb_data,
            file_name=f"resplan_{selected_id}.glb",
            mime="model/gltf-binary",
            use_container_width=True,
        )
    with details:
        st.json(
            {
                "plan_id": result.plan_id,
                "dimensions_m": result.metadata()["dimensions_m"],
                "scale_factor": result.scale_factor,
                "warnings": result.warnings,
                "options": asdict(options),
                "door_treatments": result.door_treatments,
                "geometry_variations": result.geometry_variations,
            },
            expanded=False,
        )


main()
