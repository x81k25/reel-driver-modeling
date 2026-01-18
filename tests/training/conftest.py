"""
Training-specific conftest.py

This file exists to provide fixtures for training tests without
importing the app module (which requires env vars).
Training unit tests are self-contained and don't need API fixtures.
"""
import pytest


@pytest.fixture
def project_root():
    """Return the project root directory."""
    import os
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
