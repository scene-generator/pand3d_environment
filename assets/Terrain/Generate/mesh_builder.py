from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Any, Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import (
    Polygon,
    MultiPolygon,
    LineString,
    MultiLineString,
    GeometryCollection,
)
from shapely.plotting import plot_polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import triangulate
from shapely.ops import unary_union
from shapely.geometry.polygon import orient
from panda3d.core import CullFaceAttrib
import trimesh
import matplotlib.pyplot as plt

from panda3d.core import (
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    GeomVertexReader,
    NodePath,
    Filename,
    Texture,
    TexturePool,
    TextureStage,
)

ROAD_WIDTHS = {
    "motorway": 22.0,
    "trunk": 18.0,
    "primary": 14.0,
    "secondary": 10.0,
    "tertiary": 8.0,
    "residential": 6.0,
    "service": 4.0,
    "unclassified": 5.0,
    "living_street": 4.5,
    "default": 5.0,
}

DEFAULT_BUILDING_HEIGHT = 8.0
DEFAULT_ROAD_THICKNESS = -3.0
DEFAULT_WATER_THICKNESS = -3.0
DEFAULT_PARK_THICKNESS = -3.0
BRIDGE_CLEARANCE = 5.0
EPS = 1e-8
DEFAULT_SIDEWALK_WIDTH = 1.5
DEFAULT_SIDEWALK_THICKNESS = -2.85
SIDEWALK_GAP = 0.5
DEFAULT_SURFACE_THICKNESS = -3.0
VERIFY = False


def _sidewalk_side(row, side: str) -> bool:
    """
    Determine whether an OSM road has a sidewalk on a given side.

    Supported:
        sidewalk=both
        sidewalk=left
        sidewalk=right
        sidewalk=yes
        sidewalk=no

    Also supports:
        sidewalk:left
        sidewalk:right
    """
    sidewalk = str(row.get("sidewalk", "")).strip().lower()

    if sidewalk == "both":
        return True

    if sidewalk == side:
        return True

    if sidewalk == "yes":
        return True

    if sidewalk == "no":
        return False

    side_value = str(row.get(f"sidewalk:{side}", "")).strip().lower()

    return side_value in ("yes", "true", "1")


def _first_scalar(value, default=None):
    if value is None:
        return default
    if isinstance(value, (list, tuple, set)):
        for v in value:
            return v
        return default
    return value


def _normalize_highway(value) -> str:
    value = _first_scalar(value, "default")
    if value is None:
        return "default"
    return str(value).split(";")[0].strip().lower()


def _parse_height_m(value) -> Optional[float]:
    if value is None:
        return None
    try:
        s = str(value).strip().lower()
        s = s.replace("meters", "").replace("meter", "").replace("m", "").strip()
        return float(s)
    except Exception:
        return None


def _representative_xy(geom: BaseGeometry) -> tuple[float, float]:
    if geom is None or geom.is_empty:
        return 0.0, 0.0
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        pt = geom.representative_point()
    elif geom.geom_type in ("LineString", "MultiLineString"):
        pt = geom.interpolate(0.5, normalized=True)
    elif geom.geom_type == "Point":
        pt = geom
    else:
        pt = geom.centroid
    return float(pt.x), float(pt.y)


def _ensure_single_polygon(geom: BaseGeometry):
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "Polygon":
        return geom
    if geom.geom_type == "MultiPolygon":
        polys = list(geom.geoms)
        if not polys:
            return None
        return max(polys, key=lambda p: p.area)
    return None


def _buffer_road_geometry(geom: BaseGeometry, width: float):
    if geom is None or geom.is_empty:
        return None

    if geom.geom_type in ("LineString", "MultiLineString"):
        return geom.buffer(
            width * 0.5,
            cap_style=2,
            join_style=1,
            resolution=4,
        )

    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom

    return None


def _geom_to_xy_polygon(geom: BaseGeometry):
    poly = _ensure_single_polygon(geom)
    if poly is not None:
        return poly
    return None


def _polygon_valid_tris(poly: Polygon):
    tris = triangulate(poly)
    valid = []
    if VERIFY:
        fig, ax = plt.subplots()
    for tri in tris:
        rp = tri.representative_point()
        if poly.contains(rp) or poly.touches(rp):
            valid.append(tri)
        if VERIFY:
            ax.plot(tri.exterior.xy[0], tri.exterior.xy[1], "-o")
            ax.fill(tri.exterior.xy[0], tri.exterior.xy[1], alpha=0.5)
    return valid


def _face_normal(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    n = np.cross(b - a, c - a)
    norm = np.linalg.norm(n)
    if norm < EPS:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return n / norm


def _repair_geom(geom: BaseGeometry):
    if geom is None or geom.is_empty:
        return None
    try:
        geom = geom.buffer(0)
    except Exception:
        return None
    if geom.is_empty:
        return None
    return geom


def _iter_polygon_parts(geom: BaseGeometry):
    """
    Yield Polygon objects from Polygon, MultiPolygon, or GeometryCollection.
    """
    geom = _repair_geom(geom)
    if geom is None:
        return

    if geom.geom_type == "Polygon":
        yield orient(geom, sign=1.0)

    elif geom.geom_type == "MultiPolygon":
        for part in geom.geoms:
            part = _repair_geom(part)
            if part is not None and part.geom_type == "Polygon":
                yield orient(part, sign=1.0)

    elif geom.geom_type == "GeometryCollection":
        for part in geom.geoms:
            yield from _iter_polygon_parts(part)


def _iter_rings(poly: Polygon):
    yield poly.exterior
    for hole in poly.interiors:
        yield hole


@dataclass
class MeshAsset:
    name: str
    nodepath: NodePath
    kind: str
    source_index: int
    metadata: dict


class CityMeshBuilder:
    def __init__(
        self,
        terrain_sampler: Callable[[float, float], float],
        output_dir: str = "city_output",
    ):
        self.terrain_sampler = terrain_sampler
        self.output_dir = Path(output_dir)
        self.assets: list[MeshAsset] = []
        self.buildings: list[BuildingAsset] = []
        self.tile_size = 500.0
        self.tiles = {}
        self.surface_coverage: dict[tuple[int, int], BaseGeometry] = {}
        self.root = NodePath("city_root")
        if VERIFY:
            self.fig, self.ax = plt.subplots()

    def sample_ground_z(self, geom: BaseGeometry) -> float:
        x, y = _representative_xy(geom)
        return float(self.terrain_sampler(x, y))

    def _tile_key(self, x: float, y: float) -> tuple[int, int]:
        return (int(np.floor(x / self.tile_size)), int(np.floor(y / self.tile_size)))

    def _get_tile_node(self, tile_key: tuple[int, int]) -> NodePath:
        if tile_key not in self.tiles:
            ix, iy = tile_key
            tile_node = self.root.attachNewNode(f"tile_{ix}_{iy}")
            self.tiles[tile_key] = tile_node
        return self.tiles[tile_key]

    def _claim_surface_geometry(
        self, geom: BaseGeometry, tile_key: tuple[int, int]
    ) -> BaseGeometry | None:
        """Remove previously claimed surface area, then add the remainder."""
        geom = _repair_geom(geom)
        if geom is None:
            return None

        covered = self.surface_coverage.get(tile_key)
        if covered is not None:
            geom = _repair_geom(geom.difference(covered))
            if geom is None:
                return None

        self.surface_coverage[tile_key] = _repair_geom(
            geom if covered is None else unary_union((covered, geom))
        )
        return geom

    @staticmethod
    def _normalize_landuse(value) -> str | None:
        value = _first_scalar(value)
        if value is None or pd.isna(value):
            return None
        return str(value).split(";")[0].strip().lower() or None

    def _trimesh_to_panda(
        self,
        mesh: trimesh.Trimesh,
        name: str,
    ) -> NodePath:
        """
        Convert a trimesh.Trimesh into a Panda3D GeomNode.
        """

        fmt = GeomVertexFormat.getV3n3t2()

        vdata = GeomVertexData(
            name,
            fmt,
            Geom.UHStatic,
        )

        vwriter = GeomVertexWriter(
            vdata,
            "vertex",
        )

        nwriter = GeomVertexWriter(
            vdata,
            "normal",
        )

        twriter = GeomVertexWriter(
            vdata,
            "texcoord",
        )

        # ---------------------------------------------------------
        # Vertices, normals, UVs
        # ---------------------------------------------------------
        vertices = np.asarray(mesh.vertices)
        normals = np.asarray(mesh.vertex_normals)

        for vertex, normal in zip(vertices, normals):

            x = float(vertex[0])
            y = float(vertex[1])
            z = float(vertex[2])

            vwriter.addData3f(
                x,
                y,
                z,
            )

            nwriter.addData3f(
                float(normal[0]),
                float(normal[1]),
                float(normal[2]),
            )

            # Simple world-space UVs.
            # Adjust these values to control texture repetition.
            twriter.addData2f(
                x * 0.02,
                y * 0.02,
            )

        # ---------------------------------------------------------
        # Triangle indices
        # ---------------------------------------------------------
        prim = GeomTriangles(Geom.UHStatic)

        for face in mesh.faces:

            prim.addVertices(
                int(face[0]),
                int(face[1]),
                int(face[2]),
            )

            prim.closePrimitive()

        # ---------------------------------------------------------
        # Panda3D Geom
        # ---------------------------------------------------------
        geom = Geom(vdata)
        geom.addPrimitive(prim)

        node = GeomNode(name)
        node.addGeom(geom)

        nodepath = NodePath(node)

        return nodepath

    def _extrude_to_trimesh(
        self,
        geom: BaseGeometry,
        height: float,
    ):
        geom = _repair_geom(geom)

        if geom is None:
            return None

        meshes = []

        for poly in _iter_polygon_parts(geom):
            try:
                mesh = trimesh.creation.extrude_polygon(
                    poly,
                    height=float(height),
                )

                if not mesh.is_empty:
                    meshes.append(mesh)

            except Exception as e:
                print(f"[WARN] Failed to extrude geometry: {e}")

        if not meshes:
            return None

        return meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)

    def _make_geomnode_from_extrusion(
        self,
        geom: BaseGeometry,
        height: float,
        name: str,
    ) -> NodePath:
        """
        Convert:

            Shapely geometry
                -> trimesh extrusion
                -> Panda3D GeomNode

        Supports Polygon, MultiPolygon, and GeometryCollection.
        """

        geom = _repair_geom(geom)

        if geom is None:
            return NodePath(name)

        meshes = []

        # ---------------------------------------------------------
        # Shapely -> trimesh extrusion
        # ---------------------------------------------------------
        for poly in _iter_polygon_parts(geom):

            if poly is None or poly.is_empty:
                continue
            try:
                mesh = trimesh.creation.extrude_polygon(
                    poly,
                    height=float(height),
                )
                if mesh.is_empty:
                    continue
                meshes.append(mesh)

            except Exception as e:
                print(f"[WARN] Failed to extrude {name}: {e}")

        if not meshes:
            return NodePath(name)

        # ---------------------------------------------------------
        # Combine polygon parts
        # ---------------------------------------------------------
        if len(meshes) == 1:
            mesh = meshes[0]
        else:
            mesh = trimesh.util.concatenate(meshes)

        # ---------------------------------------------------------
        # trimesh -> Panda3D
        # ---------------------------------------------------------
        return self._trimesh_to_panda(
            mesh,
            name,
        )

    def _add_asset(
        self,
        name: str,
        nodepath: NodePath,
        kind: str,
        source_index: int,
        metadata: dict,
        tile_key,
    ):
        parent = self._get_tile_node(tile_key)
        nodepath.reparentTo(parent)
        self.assets.append(
            MeshAsset(
                name=name,
                nodepath=nodepath,
                kind=kind,
                source_index=source_index,
                metadata=metadata,
            )
        )

    def add_buildings(self, buildings: gpd.GeoDataFrame):
        if buildings is None or buildings.empty:
            return

        # Optional small cache if many buildings hit the same point
        ground_cache = {}

        asset_start = len(self.assets)

        for asset_idx, row in enumerate(buildings.itertuples(index=False)):
            geom = getattr(row, "geometry", None)
            geom = _repair_geom(geom)
            if geom is None or not any(_iter_polygon_parts(geom)):
                continue

            # Use precomputed metadata if present
            height = getattr(row, "height_m", None)
            if type(height) == str:
                try:
                    height = float(height)
                except ValueError:
                    height = None
            if height is None or np.isnan(height):
                height = _parse_height_m(getattr(row, "height", None))
            if height is None or np.isnan(height):
                levels = getattr(row, "building_levels", None)
                if levels is None:
                    levels = getattr(
                        row, "_4", None
                    )  # only if your dataframe layout is odd; otherwise remove
                try:
                    if levels is not None and not pd.isna(levels):
                        height = float(levels) * 3.0
                except Exception:
                    pass
            if height is None or np.isnan(height):
                height = DEFAULT_BUILDING_HEIGHT
            x, y = _representative_xy(geom)
            tile_key = self._tile_key(x, y)

            # Cache ground sampling by tile center-ish location
            cache_key = (round(x, 2), round(y, 2))
            if cache_key in ground_cache:
                base_z = ground_cache[cache_key]
            else:
                base_z = self.sample_ground_z(geom)
                ground_cache[cache_key] = base_z

            landuse_tag = self._normalize_landuse(getattr(row, "landuse_tag", None))
            if landuse_tag is None:
                landuse_tag = self._normalize_landuse(getattr(row, "building", None))

            building_name = f"building_{asset_start + asset_idx}"
            node = self._make_geomnode_from_extrusion(
                geom,
                height,
                building_name,
            )
            color = tuple(random.uniform(0.2, 1.0) for _ in range(3))
            node.setColor(
                color[0],
                color[1],
                color[2],
                1.0,
            )
            node.setZ(base_z)
            node.setAttrib(CullFaceAttrib.make(CullFaceAttrib.MCullNone))
            building = BuildingAsset(
                name=building_name,
                nodepath=node,
                source_index=asset_idx,
                height_m=float(height),
                base_z=float(base_z),
                landuse_tag=landuse_tag,
                building_tag=getattr(row, "building", None),
                metadata={
                    "height_m": float(height),
                    "base_z": float(base_z),
                    "landuse_tag": landuse_tag,
                    "building": getattr(row, "building", None),
                },
            )

            parent = self._get_tile_node(tile_key)
            node.reparentTo(parent)

            self.buildings.append(building)
            self.assets.append(
                MeshAsset(
                    name=building.name,
                    nodepath=building.nodepath,
                    kind="building",
                    source_index=asset_idx,
                    metadata=building.metadata,
                )
            )
            # return

    def add_roads(self, roads: gpd.GeoDataFrame):
        if roads is None or roads.empty:
            return

        # ---------------------------------------------------------
        # Group road polygons by tile
        # ---------------------------------------------------------
        road_polygons = {}
        bridge_polygons = {}

        road_z_values = {}
        bridge_z_values = {}

        road_counts = {}
        bridge_counts = {}

        road_columns = set(roads.columns)

        for asset_idx, row in enumerate(roads.itertuples(index=False)):
            geom = row.geometry

            if geom is None or geom.is_empty:
                continue

            # -----------------------------------------------------
            # Highway type
            # -----------------------------------------------------
            if "highway" in road_columns:
                highway = _normalize_highway(getattr(row, "highway", "default"))
            else:
                highway = "default"

            width = float(
                ROAD_WIDTHS.get(
                    highway,
                    ROAD_WIDTHS["default"],
                )
            )

            # -----------------------------------------------------
            # Create road footprint
            # -----------------------------------------------------
            road_poly = _buffer_road_geometry(
                geom,
                width,
            )

            if road_poly is None or road_poly.is_empty:
                continue

            # -----------------------------------------------------
            # Bridge detection
            # -----------------------------------------------------
            is_bridge = False

            if "bridge" in road_columns:
                bridge_value = getattr(
                    row,
                    "bridge",
                    None,
                )

                is_bridge = str(bridge_value).strip().lower() in (
                    "yes",
                    "true",
                    "1",
                )

            if "layer" in road_columns:
                try:
                    layer = float(getattr(row, "layer", 0))

                    if layer > 0:
                        is_bridge = True

                except (TypeError, ValueError):
                    pass

            # -----------------------------------------------------
            # Terrain height
            # -----------------------------------------------------
            base_z = self.sample_ground_z(geom)

            if is_bridge:
                base_z += BRIDGE_CLEARANCE

            # -----------------------------------------------------
            # Tile
            # -----------------------------------------------------
            x, y = _representative_xy(geom)

            tile_key = self._tile_key(x, y)

            # -----------------------------------------------------
            # Store polygon
            # -----------------------------------------------------
            if is_bridge:

                bridge_polygons.setdefault(tile_key, []).append(road_poly)

                bridge_z_values.setdefault(tile_key, []).append(base_z)

                bridge_counts[tile_key] = bridge_counts.get(tile_key, 0) + 1

            else:

                road_polygons.setdefault(tile_key, []).append(road_poly)

                road_z_values.setdefault(tile_key, []).append(base_z)

                road_counts[tile_key] = road_counts.get(tile_key, 0) + 1

        # =========================================================
        # Merge normal roads
        # =========================================================
        for tile_key, polygons in road_polygons.items():

            if not polygons:
                continue

            merged = unary_union(polygons)

            # Repair geometry
            merged = _repair_geom(merged)

            if merged is None:
                continue

            merged = self._claim_surface_geometry(merged, tile_key)
            if merged is None:
                continue

            # -----------------------------------------------------
            # Determine elevation
            # -----------------------------------------------------
            z_values = road_z_values[tile_key]

            if z_values:
                base_z = float(np.mean(z_values))
            else:
                base_z = 0.0

            # -----------------------------------------------------
            # Shapely -> trimesh
            # -----------------------------------------------------
            mesh = self._extrude_to_trimesh(
                merged,
                DEFAULT_ROAD_THICKNESS,
            )

            if mesh is None:
                continue

            # -----------------------------------------------------
            # Move to terrain
            # -----------------------------------------------------
            mesh.apply_translation(
                [
                    0.0,
                    0.0,
                    base_z,
                ]
            )

            # -----------------------------------------------------
            # Convert to Panda3D
            # -----------------------------------------------------
            name = f"road_layer_" f"{tile_key[0]}_" f"{tile_key[1]}"

            node = self._trimesh_to_panda(
                mesh,
                name,
            )

            node.reparentTo(self._get_tile_node(tile_key))

            # -----------------------------------------------------
            # Register layer
            # -----------------------------------------------------
            self.assets.append(
                MeshAsset(
                    name=name,
                    nodepath=node,
                    kind="road",
                    source_index=-1,
                    metadata={
                        "layer": "road",
                        "tile": tile_key,
                        "feature_count": road_counts[tile_key],
                        "merged": True,
                    },
                )
            )

        # =========================================================
        # Bridges
        # =========================================================
        for tile_key, polygons in bridge_polygons.items():

            if not polygons:
                continue

            merged = unary_union(polygons)

            merged = _repair_geom(merged)

            if merged is None:
                continue

            z_values = bridge_z_values[tile_key]

            base_z = float(np.mean(z_values)) if z_values else 0.0

            mesh = self._extrude_to_trimesh(
                merged,
                DEFAULT_ROAD_THICKNESS,
            )

            if mesh is None:
                continue

            mesh.apply_translation(
                [
                    0.0,
                    0.0,
                    base_z,
                ]
            )

            name = f"bridge_layer_" f"{tile_key[0]}_" f"{tile_key[1]}"

            node = self._trimesh_to_panda(
                mesh,
                name,
            )

            node.reparentTo(self._get_tile_node(tile_key))

            self.assets.append(
                MeshAsset(
                    name=name,
                    nodepath=node,
                    kind="bridge",
                    source_index=-1,
                    metadata={
                        "layer": "bridge",
                        "tile": tile_key,
                        "feature_count": bridge_counts[tile_key],
                        "merged": True,
                    },
                )
            )

    def add_sidewalks(
        self,
        sidewalks: gpd.GeoDataFrame | None = None,
        roads: gpd.GeoDataFrame | None = None,
        width: float = DEFAULT_SIDEWALK_WIDTH,
        thickness: float = DEFAULT_SIDEWALK_THICKNESS,
    ):
        """
        Build merged sidewalk layers per tile.

        Sources:
            1. Explicit OSM sidewalk geometries
            2. Sidewalks inferred from OSM road tags

        The final output is one sidewalk mesh per tile.
        """

        # ============================================================
        # Collect sidewalk polygons by tile
        # ============================================================
        sidewalk_tiles: dict[tuple[int, int], list[BaseGeometry]] = {}
        sidewalk_counts: dict[tuple[int, int], int] = {}

        # ============================================================
        # 1. Explicitly mapped sidewalks
        # ============================================================
        if sidewalks is not None and not sidewalks.empty:

            geom_idx = sidewalks.columns.get_loc("geometry")

            for asset_idx, row in enumerate(
                sidewalks.itertuples(index=False, name=None)
            ):
                geom = row[geom_idx]

                if geom is None or geom.is_empty:
                    continue

                if geom.geom_type not in (
                    "LineString",
                    "MultiLineString",
                ):
                    continue

                sidewalk_poly = geom.buffer(
                    width * 0.5,
                    cap_style=2,
                    join_style=1,
                )

                if sidewalk_poly is None or sidewalk_poly.is_empty:
                    continue

                x, y = _representative_xy(geom)
                tile_key = self._tile_key(x, y)

                sidewalk_tiles.setdefault(
                    tile_key,
                    [],
                ).append(sidewalk_poly)

                sidewalk_counts[tile_key] = sidewalk_counts.get(tile_key, 0) + 1

        # ============================================================
        # 2. Generate sidewalks from road tags
        # ============================================================
        if roads is not None and not roads.empty:

            cols = roads.columns
            geom_idx = cols.get_loc("geometry")

            has_sidewalk = "sidewalk" in cols
            has_left = "sidewalk:left" in cols
            has_right = "sidewalk:right" in cols

            if has_sidewalk or has_left or has_right:

                for asset_idx, row in enumerate(
                    roads.itertuples(index=False, name=None)
                ):
                    geom = row[geom_idx]

                    if geom is None or geom.is_empty:
                        continue

                    if geom.geom_type not in (
                        "LineString",
                        "MultiLineString",
                    ):
                        continue

                    # ------------------------------------------------
                    # Convert tuple back to named values
                    # ------------------------------------------------
                    row_data = {col: row[i] for i, col in enumerate(cols)}

                    create_left = _sidewalk_side(
                        row_data,
                        "left",
                    )

                    create_right = _sidewalk_side(
                        row_data,
                        "right",
                    )

                    if not create_left and not create_right:
                        continue

                    # ------------------------------------------------
                    # Road width
                    # ------------------------------------------------
                    highway = _normalize_highway(
                        row_data.get(
                            "highway",
                            "default",
                        )
                    )

                    road_width = float(
                        ROAD_WIDTHS.get(
                            highway,
                            ROAD_WIDTHS["default"],
                        )
                    )

                    # Centerline -> sidewalk center
                    offset = road_width * 0.5 + width * 0.5 + SIDEWALK_GAP

                    # ------------------------------------------------
                    # Representative point / tile
                    # ------------------------------------------------
                    x, y = _representative_xy(geom)

                    tile_key = self._tile_key(
                        x,
                        y,
                    )

                    # ------------------------------------------------
                    # LEFT
                    # ------------------------------------------------
                    if create_left:

                        sidewalk_line = geom.parallel_offset(
                            offset,
                            "left",
                            join_style=1,
                        )

                        if sidewalk_line is not None and not sidewalk_line.is_empty:

                            sidewalk_poly = sidewalk_line.buffer(
                                width * 0.5,
                                cap_style=2,
                                join_style=1,
                            )

                            if sidewalk_poly is not None and not sidewalk_poly.is_empty:
                                sidewalk_tiles.setdefault(
                                    tile_key,
                                    [],
                                ).append(sidewalk_poly)

                                sidewalk_counts[tile_key] = (
                                    sidewalk_counts.get(
                                        tile_key,
                                        0,
                                    )
                                    + 1
                                )

                    # ------------------------------------------------
                    # RIGHT
                    # ------------------------------------------------
                    if create_right:

                        sidewalk_line = geom.parallel_offset(
                            offset,
                            "right",
                            join_style=1,
                        )

                        if sidewalk_line is not None and not sidewalk_line.is_empty:

                            sidewalk_poly = sidewalk_line.buffer(
                                width * 0.5,
                                cap_style=2,
                                join_style=1,
                            )

                            if sidewalk_poly is not None and not sidewalk_poly.is_empty:
                                sidewalk_tiles.setdefault(
                                    tile_key,
                                    [],
                                ).append(sidewalk_poly)

                                sidewalk_counts[tile_key] = (
                                    sidewalk_counts.get(
                                        tile_key,
                                        0,
                                    )
                                    + 1
                                )

        # ============================================================
        # Nothing to build
        # ============================================================
        if not sidewalk_tiles:
            return

        # ============================================================
        # Merge + extrude one sidewalk layer per tile
        # ============================================================
        for tile_key, polygons in sidewalk_tiles.items():

            if not polygons:
                continue

            # --------------------------------------------------------
            # Merge overlapping sidewalk polygons
            # --------------------------------------------------------
            merged = unary_union(polygons)

            merged = _repair_geom(merged)

            if merged is None:
                continue

            merged = self._claim_surface_geometry(merged, tile_key)
            if merged is None:
                continue

            # --------------------------------------------------------
            # Terrain elevation
            # --------------------------------------------------------
            base_z = self.sample_ground_z(merged)

            base_z += SIDEWALK_GAP

            # --------------------------------------------------------
            # Shapely -> trimesh
            # --------------------------------------------------------
            mesh = self._extrude_to_trimesh(
                merged,
                thickness,
            )

            if mesh is None:
                continue

            # --------------------------------------------------------
            # Move to world Z
            # --------------------------------------------------------
            mesh.apply_translation(
                [
                    0.0,
                    0.0,
                    base_z,
                ]
            )

            # --------------------------------------------------------
            # trimesh -> Panda3D
            # --------------------------------------------------------
            name = f"sidewalk_layer_" f"{tile_key[0]}_" f"{tile_key[1]}"

            node = self._trimesh_to_panda(
                mesh,
                name,
            )

            node.reparentTo(self._get_tile_node(tile_key))

            node.setAttrib(CullFaceAttrib.make(CullFaceAttrib.MCullNone))

            # --------------------------------------------------------
            # Register one asset for the entire tile
            # --------------------------------------------------------
            self.assets.append(
                MeshAsset(
                    name=name,
                    nodepath=node,
                    kind="sidewalk",
                    source_index=-1,
                    metadata={
                        "layer": "sidewalk",
                        "tile": tile_key,
                        "width_m": float(width),
                        "thickness_m": float(thickness),
                        "feature_count": sidewalk_counts.get(
                            tile_key,
                            0,
                        ),
                        "merged": True,
                    },
                )
            )

    def add_bridges(self, roads: gpd.GeoDataFrame):
        if roads is None or roads.empty:
            return

        if "bridge" not in roads.columns and "layer" not in roads.columns:
            return

        bridge_mask = np.zeros(len(roads), dtype=bool)

        if "bridge" in roads.columns:
            bridge_mask |= (
                roads["bridge"]
                .fillna("")
                .astype(str)
                .str.lower()
                .isin(["yes", "true", "1"])
                .to_numpy()
            )

        if "layer" in roads.columns:
            try:
                bridge_mask |= roads["layer"].fillna(0).astype(float).to_numpy() > 0
            except Exception:
                pass

        bridge_roads = roads.loc[bridge_mask]
        if bridge_roads.empty:
            return

        for i, row in bridge_roads.iterrows():
            geom = row.geometry
            highway = _normalize_highway(row.get("highway", "default"))
            width = float(ROAD_WIDTHS.get(highway, ROAD_WIDTHS["default"]))
            road_poly = _buffer_road_geometry(geom, width)
            if road_poly is None:
                continue

            base_z = self.sample_ground_z(geom) + BRIDGE_CLEARANCE
            x, y = _representative_xy(geom)
            tile_key = self._tile_key(x, y)
            node = self._make_geomnode_from_extrusion(
                road_poly,
                DEFAULT_ROAD_THICKNESS,
                f"bridge_{len(self.assets)}",
            )
            node.setZ(base_z)
            node.setAttrib(CullFaceAttrib.make(CullFaceAttrib.MCullNone))
            node.setTwoSided(True)

            self._add_asset(
                name=f"bridge_{len(self.assets)}",
                nodepath=node,
                kind="bridge",
                source_index=int(i),
                metadata={
                    "highway": highway,
                    "width_m": width,
                    "base_z": float(base_z),
                    "bridge": True,
                },
                tile_key=tile_key,
            )

    def add_water(self, water: gpd.GeoDataFrame):
        if water is None or water.empty:
            return
        for asset_idx, (gdf_idx, row) in enumerate(water.iterrows()):
            poly = _geom_to_xy_polygon(row.geometry)
            if poly is None:
                continue
            x, y = _representative_xy(poly)
            tile_key = self._tile_key(x, y)
            poly = self._claim_surface_geometry(poly, tile_key)
            if poly is None:
                continue
            z = self.sample_ground_z(poly)
            node = self._make_geomnode_from_extrusion(
                poly,
                DEFAULT_WATER_THICKNESS,
                f"water_{len(self.assets)}",
            )
            node.setZ(z)
            node.setAttrib(CullFaceAttrib.make(CullFaceAttrib.MCullNone))
            node.setTwoSided(True)
            self._add_asset(
                name=f"water_{len(self.assets)}",
                nodepath=node,
                kind="water",
                source_index=asset_idx,
                metadata={"surface_z": float(z)},
                tile_key=tile_key,
            )

    def add_parks(self, parks: gpd.GeoDataFrame):
        if parks is None or parks.empty:
            return
        for asset_idx, (gdf_idx, row) in enumerate(parks.iterrows()):
            poly = _geom_to_xy_polygon(row.geometry)
            if poly is None:
                continue
            x, y = _representative_xy(poly)
            tile_key = self._tile_key(x, y)
            poly = self._claim_surface_geometry(poly, tile_key)
            if poly is None:
                continue
            z = self.sample_ground_z(poly)
            node = self._make_geomnode_from_extrusion(
                poly,
                DEFAULT_PARK_THICKNESS,
                f"park_{len(self.assets)}",
            )
            node.setZ(z)
            node.setAttrib(CullFaceAttrib.make(CullFaceAttrib.MCullNone))
            node.setTwoSided(True)
            self._add_asset(
                name=f"park_{len(self.assets)}",
                nodepath=node,
                kind="park",
                source_index=asset_idx,
                metadata={"surface_z": float(z)},
                tile_key=tile_key,
            )

    def add_surface_features(self, features: gpd.GeoDataFrame | None, kind: str):
        if features is None or features.empty:
            return

        z_offsets = {
            "terrain": -0.08,
            "landuse": -0.02,
            "parking": 0.03,
            "railway_areas": 0.035,
            "pedestrian": 0.045,
        }
        if kind not in z_offsets:
            raise ValueError(f"Unsupported ground surface kind: {kind}")

        for asset_idx, (_, row) in enumerate(features.iterrows()):
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            surface_geom = geom
            if kind == "pedestrian" and geom.geom_type in (
                "LineString",
                "MultiLineString",
            ):
                highway = _normalize_highway(row.get("highway", "default"))
                width = 3.0 if highway == "pedestrian" else 2.0
                surface_geom = geom.buffer(width * 0.5, cap_style=2, join_style=2)
            if surface_geom.geom_type not in (
                "Polygon",
                "MultiPolygon",
                "GeometryCollection",
            ):
                continue
            surface_geom = _repair_geom(surface_geom)
            if surface_geom is None:
                continue

            feature_tag = None
            for column in (
                "landuse",
                "amenity",
                "leisure",
                "highway",
                "railway",
                "natural",
                "landcover",
            ):
                value = _first_scalar(row.get(column))
                if value is not None and not pd.isna(value):
                    feature_tag = str(value).split(";")[0].strip().lower()
                    break

            x, y = _representative_xy(surface_geom)
            tile_key = self._tile_key(x, y)
            surface_geom = self._claim_surface_geometry(surface_geom, tile_key)
            if surface_geom is None:
                continue
            z = self.sample_ground_z(surface_geom)
            name = f"{kind}_{len(self.assets)}"
            node = self._make_geomnode_from_extrusion(
                surface_geom,
                DEFAULT_SURFACE_THICKNESS,
                name,
            )
            node.setZ(z + z_offsets[kind])
            node.setTwoSided(True)
            self._add_asset(
                name=name,
                nodepath=node,
                kind=kind,
                source_index=asset_idx,
                metadata={
                    "feature_tag": feature_tag,
                    "surface_z": float(z + z_offsets[kind]),
                    "thickness_m": DEFAULT_SURFACE_THICKNESS,
                    "source": "osm",
                },
                tile_key=tile_key,
            )

    def _write_obj_from_nodepath(
        self, nodepath: NodePath, fh, object_name: str, vertex_offset: int
    ) -> int:
        node = nodepath.node()
        if isinstance(node, GeomNode):
            geom_paths = [nodepath]
        else:
            geom_paths = list(nodepath.findAllMatches("**/+GeomNode"))
        if not geom_paths:
            return vertex_offset

        fh.write(f"o {object_name}\n")
        fh.write(f"g {object_name}\n")

        for geom_path in geom_paths:
            geom_node = geom_path.node()
            transform = geom_path.getMat(self.root)
            for geom_index in range(geom_node.getNumGeoms()):
                geom = geom_node.getGeom(geom_index)
                vdata = geom.getVertexData()
                vreader = GeomVertexReader(vdata, "vertex")

                nreader = None
                try:
                    nreader = GeomVertexReader(vdata, "normal")
                except Exception:
                    nreader = None

                local_vertices = []
                local_normals = []

                while not vreader.isAtEnd():
                    vertex = transform.xformPoint(vreader.getData3f())
                    local_vertices.append(
                        (float(vertex[0]), float(vertex[1]), float(vertex[2]))
                    )

                    if nreader is not None and not nreader.isAtEnd():
                        normal = nreader.getData3f()
                        local_normals.append(
                            (float(normal[0]), float(normal[1]), float(normal[2]))
                        )

                for vx, vy, vz in local_vertices:
                    fh.write(f"v {vx:.6f} {vy:.6f} {vz:.6f}\n")
                for nx, ny, nz in local_normals:
                    fh.write(f"vn {nx:.6f} {ny:.6f} {nz:.6f}\n")

                for prim_index in range(geom.getNumPrimitives()):
                    prim = geom.getPrimitive(prim_index)
                    for primitive_index in range(prim.getNumPrimitives()):
                        start = prim.getPrimitiveStart(primitive_index)
                        end = prim.getPrimitiveEnd(primitive_index)
                        if end - start != 3:
                            continue
                        i0 = prim.getVertex(start) + 1 + vertex_offset
                        i1 = prim.getVertex(start + 1) + 1 + vertex_offset
                        i2 = prim.getVertex(start + 2) + 1 + vertex_offset
                        fh.write(f"f {i0} {i1} {i2}\n")

                vertex_offset += len(local_vertices)

        return vertex_offset

    def _iter_polygon_rings(self, geom):
        if geom is None or geom.is_empty:
            return

        if geom.geom_type == "Polygon":
            yield geom

        elif geom.geom_type == "MultiPolygon":
            for part in geom.geoms:
                yield part

        elif geom.geom_type == "GeometryCollection":
            for part in geom.geoms:
                yield from self._iter_polygon_rings(part)

    def export_bam(self, filename: str = "city.bam", flatten: bool = False) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = Path(self.output_dir, filename)

        # if flatten:
        #     self.root.flattenStrong()
        self.root.writeBamFile(Filename.fromOsSpecific(str(out_path.resolve())))
        return out_path

    def export_obj(self, filename: str = "city.obj") -> Path:
        """Export the city as a Wavefront OBJ file."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / filename

        with out_path.open("w", encoding="utf-8") as fh:
            fh.write("# City mesh export\n")
            fh.write(f"# assets={len(self.assets)}\n")
            fh.write("mtllib city.mtl\n")

            vertex_offset = 0
            for asset in self.assets:
                vertex_offset = self._write_obj_from_nodepath(
                    asset.nodepath,
                    fh,
                    asset.name,
                    vertex_offset,
                )

        mtl_path = self.output_dir / "city.mtl"
        with mtl_path.open("w", encoding="utf-8") as fh:
            fh.write("newmtl default\n")
            fh.write("Kd 0.8 0.8 0.8\n")

        return out_path

    def _load_texture(self, texture_path: str) -> Texture:
        path_cwd = Path.cwd()
        texture_path = Path(path_cwd, texture_path)
        tex = TexturePool.loadTexture(texture_path)
        if tex is None:
            raise FileNotFoundError(f"Could not load texture: {texture_path}")
        return tex

    def _building_texture_for_height(
        self,
        height_m: float,
        landuse: str | None,
        texture_rules: dict,
    ):
        """
        Select a building texture.

        Parameters
        ----------
        height_m : float
            Building height in meters.

        landuse : str
            OSM landuse tag associated with the building
            (e.g. residential, commercial, retail, industrial).

        texture_rules : dict
            Example:

            {
                "residential": "textures/buildings/house.jpg",
                "commercial":  "textures/buildings/commercial.jpg",
                "retail":      "textures/buildings/retail.jpg",
                "industrial":  "textures/buildings/industrial.jpg",
                "default": [
                    (10, "textures/buildings/lowrise.jpg"),
                    (25, "textures/buildings/midrise.jpg"),
                    (1e9, "textures/buildings/highrise.jpg"),
                ]
            }
        """

        # First try the landuse texture
        if landuse:
            landuse = landuse.lower()
            if landuse in texture_rules:
                return texture_rules[landuse]

        # Fall back to height-based texture selection
        for height_max, tex_path in texture_rules["default"]:
            if height_m <= height_max:
                return tex_path

        return texture_rules["default"][-1][1]

    def apply_textures(
        self,
        building_rules,
        road_texture_path: str,
        park_texture_path: str,
        water_texture_path: str,
        sidewalk_texture_path: str | None = None,
    ):
        road_tex = self._load_texture(str(road_texture_path))
        park_tex = self._load_texture(str(park_texture_path))
        water_tex = self._load_texture(str(water_texture_path))
        sidewalk_tex = self._load_texture(str(sidewalk_texture_path))

        tex_stage = TextureStage("tex")

        texture_library = BuildingTextureLibrary(
            landuse_textures={k.lower(): Path(v) for k, v in building_rules},
            default_height_rules=[
                (
                    10.0,
                    Path(
                        "assets/textures/building_materials/bricks/Bricks101_1K-JPG/Bricks101_1K-JPG_Color.jpg"
                    ),
                ),
                (
                    25.0,
                    Path(
                        "assets/textures/building_materials/bricks/Bricks097_1K-JPG/Bricks097_1K-JPG_Color.jpg"
                    ),
                ),
                (
                    1e9,
                    Path(
                        "assets/textures/building_materials/office/window-pattern-textures-building.jpg"
                    ),
                ),
            ],
        )

        for asset in self.assets:
            if asset.kind == "road":
                asset.nodepath.setTexture(tex_stage, road_tex, 1)
            elif asset.kind == "park":
                asset.nodepath.setTexture(tex_stage, park_tex, 1)
            elif asset.kind == "water":
                asset.nodepath.setTexture(tex_stage, water_tex, 1)
            elif asset.kind == "bridge":
                asset.nodepath.setTexture(tex_stage, road_tex, 1)
            elif asset.kind == "sidewalk":
                asset.nodepath.setTexture(tex_stage, sidewalk_tex, 1)
            elif asset.kind == "pedestrian":
                asset.nodepath.setTexture(tex_stage, sidewalk_tex, 1)
            elif asset.kind in ("parking", "railway_areas"):
                asset.nodepath.setTexture(tex_stage, road_tex, 1)
            elif asset.kind in ("landuse", "terrain"):
                feature_tag = asset.metadata.get("feature_tag")
                if asset.kind == "landuse" and feature_tag in (
                    "commercial",
                    "construction",
                    "industrial",
                    "retail",
                    "residential",
                ):
                    asset.nodepath.setTexture(tex_stage, road_tex, 1)
                else:
                    asset.nodepath.setTexture(tex_stage, park_tex, 1)

    def export_manifest_csv(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for asset in self.assets:
            rec = {
                "name": asset.name,
                "kind": asset.kind,
                "source_index": asset.source_index,
            }
            rec.update(asset.metadata)
            records.append(rec)
        df = pd.DataFrame.from_records(records)
        out_path = self.output_dir / "city_manifest.csv"
        df.to_csv(out_path, index=False)
        return out_path


@dataclass
class BuildingAsset:
    name: str
    nodepath: NodePath
    source_index: int
    height_m: float
    base_z: float
    landuse_tag: Optional[str] = None
    building_tag: Optional[str] = None
    building_levels: Optional[float] = None
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_row(
        cls,
        name: str,
        nodepath: NodePath,
        source_index: int,
        row,
        height_m: float,
        base_z: float,
    ):
        def _first(v):
            if v is None:
                return None
            if isinstance(v, (list, tuple, set)):
                return next(iter(v), None)
            return v

        landuse_tag = _first(row.get("landuse_tag", None))
        building_tag = _first(row.get("building", None))
        levels = row.get("building:levels", None)

        try:
            if levels is not None and pd.isna(levels):
                levels = None
        except Exception:
            pass

        meta = dict(row.drop(labels=["geometry"], errors="ignore"))
        meta["height_m"] = float(height_m)
        meta["base_z"] = float(base_z)
        meta["landuse_tag"] = landuse_tag
        meta["building_tag"] = building_tag
        meta["building_levels"] = levels

        return cls(
            name=name,
            nodepath=nodepath,
            source_index=source_index,
            height_m=float(height_m),
            base_z=float(base_z),
            landuse_tag=landuse_tag,
            building_tag=building_tag,
            building_levels=levels,
            metadata=meta,
        )

    @property
    def texture_key(self) -> str:
        if self.landuse_tag:
            return str(self.landuse_tag).strip().lower()
        return "default"


@dataclass
class BuildingTextureLibrary:
    landuse_textures: dict[str, Path]
    default_height_rules: list[tuple[float, Path]]

    def select(self, building: BuildingAsset) -> Path:
        landuse = (building.landuse_tag or "").strip().lower()
        if landuse in self.landuse_textures:
            return self.landuse_textures[landuse]

        for height_max, tex_path in self.default_height_rules:
            if building.height_m <= height_max:
                return tex_path

        return self.default_height_rules[-1][1]
