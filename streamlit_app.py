"""Streamlit Cloud entrypoint — adds src/ to path then runs the dashboard."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from dashboard import app  # noqa: F401  — executes the app module
