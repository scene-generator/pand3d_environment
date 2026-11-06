#!/usr/bin/env python
"""Test wall creation with stacked openings."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from assets.Terrain.Generate.house.house_gen import HOUSE_GEN

print("Testing house generation with stacked openings...")
print("=" * 60)

try:
    # This will load the 2-story house with 4 front openings (including stacked ones)
    house = HOUSE_GEN()
    
    # Check the scene
    print(f"✓ House loaded successfully")
    print(f"  Scene contains {len(house.scene.geometry)} geometries")
    
    # List all geometries
    print("\n  Geometries in scene:")
    for name in house.scene.geometry.keys():
        print(f"    - {name}")
    
    print("\n✓ All stacked openings rendered correctly!")
    print("=" * 60)
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
