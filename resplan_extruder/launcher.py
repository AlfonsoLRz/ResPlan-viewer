"""Console launcher for the Streamlit viewer."""

from __future__ import annotations

from pathlib import Path
import sys


def main() -> None:
    from streamlit.web import cli as streamlit_cli

    viewer = Path(__file__).with_name("viewer.py")
    sys.argv = ["streamlit", "run", str(viewer), *sys.argv[1:]]
    raise SystemExit(streamlit_cli.main())
