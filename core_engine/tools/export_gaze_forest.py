#!/usr/bin/env python
"""Kept for the old command name:  python core_engine/tools/export_gaze_forest.py
It now exports BOTH forests (gaze + hand) - see export_forests.py for the options."""
import os
import runpy
import sys

runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "export_forests.py"), run_name="__main__")
