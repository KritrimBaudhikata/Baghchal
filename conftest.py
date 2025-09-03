"""
Pytest configuration file for Baghchal AlphaZero tests.
This file helps pytest find modules in the parent directory.
"""

import sys
import os

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
