"""
Shared pytest fixtures.
"""

import sys
import os

# Ensure src/ is importable from project root without local install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
