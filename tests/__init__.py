"""Test suite for Genetic Coil Arena."""

import pytest
import numpy as np
from pathlib import Path

# Test configuration
TEST_DATA_DIR = Path(__file__).parent / "data"
TEMP_DIR = Path(__file__).parent / "temp"

# Ensure temp directory exists
TEMP_DIR.mkdir(exist_ok=True)
