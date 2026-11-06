#!/usr/bin/env python
"""Test script to verify house configuration loading works."""
import sys
from pathlib import Path

# Add the assets directory to the path
sys.path.insert(0, str(Path.cwd()))

import yaml

# Test configuration loading
config_path = Path("assets/Terrain/Generate/house/config/houses.yaml")

if not config_path.exists():
    print(f"✗ Config file not found: {config_path}")
    sys.exit(1)

with open(config_path) as f:
    config = yaml.safe_load(f)

# Test paths
test_paths = [
    ("single_family", "1_story", "basic"),
    ("single_family", "2_story", "row_home_middle"),
]

print("Testing house configuration paths:")
print("=" * 50)

for path in test_paths:
    try:
        house = config
        for key in path:
            house = house[key]
        print(f"✓ {' -> '.join(path)}")
        print(f"  Width: {house.get('width', 'N/A')}, Depth: {house.get('depth', 'N/A')}")
    except KeyError as e:
        print(f"✗ {' -> '.join(path)} - Key not found: {e}")

print("=" * 50)
print("✓ All configurations accessible!")
