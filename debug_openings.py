#!/usr/bin/env python
"""Debug script to test multiple opening placement."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import yaml

# Load config
config_path = Path("assets/Terrain/Generate/house/config/houses.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)

# Get the 2-story house which has 4 front openings
house = config['single_family']['2_story']['row_home_middle']
width = house['width']
depth = house['depth']
front_openings = house['front_openings']

print("=" * 60)
print(f"Testing house: width={width}, depth={depth}")
print(f"Front openings: {len(front_openings)}")
print("=" * 60)

def resolve_offset(opening, wall_width):
    """Resolve offset reference to actual coordinate."""
    offset = opening["offset"]
    
    # Old format - direct number
    if isinstance(offset, (int, float)):
        return offset
    
    # New format - reference-based
    reference = offset["reference"]
    margin = offset.get("margin", 0.0)
    
    if reference == "left":
        return margin
    elif reference == "center":
        return wall_width / 2.0
    elif reference == "right":
        return wall_width - margin
    else:
        raise ValueError(f"Unknown reference: {reference}")

# Test offset resolution
print("\nOpening positions along wall:")
for i, opening in enumerate(front_openings):
    offset = resolve_offset(opening, width)
    o_width = opening['width']
    o_height = opening['height']
    o_z = opening['z']
    o_type = opening['type']
    
    # Calculate opening bounds
    left = offset - o_width / 2
    right = offset + o_width / 2
    bottom = o_z
    top = o_z + o_height
    
    print(f"\nOpening {i} ({o_type}):")
    print(f"  Position: x={offset:.2f}")
    print(f"  Horizontal bounds: [{left:.2f}, {right:.2f}]")
    print(f"  Vertical bounds: z=[{bottom:.2f}, {top:.2f}]")
    print(f"  Size: {o_width:.2f}w x {o_height:.2f}h")

# Check for overlaps
print("\n" + "=" * 60)
print("Checking for overlaps...")
openings_data = []
for i, opening in enumerate(front_openings):
    offset = resolve_offset(opening, width)
    o_width = opening['width']
    left = offset - o_width / 2
    right = offset + o_width / 2
    openings_data.append({
        'idx': i,
        'left': left,
        'right': right,
        'offset': offset,
        'width': o_width
    })

# Check for overlaps
has_overlap = False
for i in range(len(openings_data)):
    for j in range(i+1, len(openings_data)):
        o1 = openings_data[i]
        o2 = openings_data[j]
        
        # Check if they overlap
        if o1['right'] > o2['left'] and o1['left'] < o2['right']:
            print(f"✗ OVERLAP: Opening {o1['idx']} and {o2['idx']}")
            print(f"  O1: [{o1['left']:.2f}, {o1['right']:.2f}]")
            print(f"  O2: [{o2['left']:.2f}, {o2['right']:.2f}]")
            has_overlap = True

# Check if any opening exceeds wall bounds
print(f"\nWall bounds: [0, {width}]")
for o in openings_data:
    if o['left'] < 0:
        print(f"✗ Opening {o['idx']} extends beyond left wall edge: {o['left']:.2f}")
    if o['right'] > width:
        print(f"✗ Opening {o['idx']} extends beyond right wall edge: {o['right']:.2f}")

if not has_overlap:
    print("✓ No overlaps detected")
    
print("=" * 60)
