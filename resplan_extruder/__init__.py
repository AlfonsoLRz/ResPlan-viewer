"""3D extrusion tools for ResPlan floor plans."""

from .core import ExtrusionOptions, ExtrusionResult, extrude_plan
from .exporters import export_bytes, export_plan
from .loader import ensure_dataset, load_dataset, load_splits, select_plans

__all__ = [
    "ExtrusionOptions",
    "ExtrusionResult",
    "extrude_plan",
    "export_bytes",
    "export_plan",
    "ensure_dataset",
    "load_dataset",
    "load_splits",
    "select_plans",
]

__version__ = "0.1.0"
