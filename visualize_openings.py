#!/usr/bin/env python
"""Visual test to show the stacked openings issue."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import yaml

config_path = Path("assets/Terrain/Generate/house/config/houses.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)

house = config['single_family']['2_story']['row_home_middle']
width = house['width']
front_openings = house['front_openings']

def resolve_offset(opening, wall_width):
    offset = opening["offset"]
    if isinstance(offset, (int, float)):
        return offset
    reference = offset["reference"]
    margin = offset.get("margin", 0.0)
    if reference == "left":
        return margin
    elif reference == "center":
        return wall_width / 2.0
    elif reference == "right":
        return wall_width - margin

print("\nWall ASCII visualization (front wall, width=5.0, height=6.7):")
print("=" * 60)

# Create a 2D grid to visualize
wall_height = 6.7
grid_width = 50  # Characters
grid_height = 20  # Characters

grid = [['.' for _ in range(grid_width)] for _ in range(grid_height)]

# Place each opening on the grid
for i, opening in enumerate(front_openings):
    offset = resolve_offset(opening, width)
    o_width = opening['width']
    o_height = opening['height']
    o_z = opening['z']
    
    # Convert to grid coordinates
    x_min = int((offset - o_width/2) / width * grid_width)
    x_max = int((offset + o_width/2) / width * grid_width)
    z_min = int(o_z / wall_height * grid_height)
    z_max = int((o_z + o_height) / wall_height * grid_height)
    
    x_min = max(0, min(grid_width-1, x_min))
    x_max = max(0, min(grid_width-1, x_max))
    z_min = max(0, min(grid_height-1, z_min))
    z_max = max(0, min(grid_height-1, z_max))
    
    char = chr(ord('A') + i)
    for z in range(z_min, z_max + 1):
        for x in range(x_min, x_max + 1):
            if grid[grid_height - 1 - z][x] == '.':
                grid[grid_height - 1 - z][x] = char
            else:
                # Mark overlaps
                grid[grid_height - 1 - z][x] = 'X'

# Print the grid
for row in grid:
    print(''.join(row))
print("=" * 60)
print("\nLegend:")
for i, opening in enumerate(front_openings):
    char = chr(ord('A') + i)
    print(f"  {char} = Opening {i} ({opening['type']})")
print("  . = Wall")
print("  X = Overlap/Conflict!")
