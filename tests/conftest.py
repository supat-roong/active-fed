"""
Shared pytest fixtures.
"""

import os
import sys

# Ensure src/ is importable from project root without local install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
