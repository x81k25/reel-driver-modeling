import pytest
import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Set development environment for tests
os.environ.setdefault('LOCAL_DEVELOPMENT', 'true')
