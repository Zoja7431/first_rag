"""
src/ — main package
"""
import sys
from pathlib import Path

# Авто sys.path для импортов from src.* из tests/root
sys.path.insert(0, str(Path(__file__).parent))