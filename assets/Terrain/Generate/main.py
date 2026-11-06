from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd
from shapely.geometry import box

from mesh_builder import CityMeshBuilder
import mesh_builder as mb
from osm import OSMDownloader
from terrain import ConstantTerrainSampler, RasterTerrainSampler


def _parse_point(value: str) -> tuple[float, float]:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Point must be formatted as 'lat,lon'")
    return float(parts[0].strip()), float(parts[1].strip())


def _location_key(place: str | None, point: tuple[float, float] | None) -> str:
    if place:
        return place
    if point is None:
        raise ValueError("Either place or point must be provided")
    return f"point_{point[0]}_{point[1]}"


def save_gdf(gdf: gpd.GeoDataFrame | None, path: Path) -> Path | None:
    if gdf is None or gdf.empty:
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        out = path.with_suffix(".parquet")
        gdf.to_parquet(out)
        return out
    except Exception:
        out = path.with_suffix(".geojson")
        gdf.to_file(out, driver="GeoJSON")
        return out


def _load_cached_gdf(path: Path) -> gpd.GeoDataFrame | None:
    if path.suffix.lower() == ".parquet" and path.exists():
        return gpd.read_parquet(path)
    if path.suffix.lower() in (".geojson", ".json") and path.exists():
        return gpd.read_file(path)
    return None


def _query_gdf_layer(
    osm: OSMDownloader,
    layer_name: str,
    place: str | None,
    point: tuple[float, float] | None,
    use_cache: bool,
    add_elevation: bool,
) -> tuple[gpd.GeoDataFrame | None, str]:
    key = _location_key(place, point)
    stem = osm._gdf_cache_stem(layer_name, key)

    if use_cache:
        cached = osm._load_gdf(stem)
        if cached is not None:
            return cached, stem

    if place:
        query = getattr(osm, f"{layer_name}_from_place")
        gdf = query(
            place, add_elevation=add_elevation, use_cache=use_cache, save_cache=True
        )
    else:
        query = getattr(osm, f"{layer_name}_from_point")
        gdf = query(
            point,
            add_elevation=add_elevation,
            use_cache=False,
            save_cache=False,
            cache_name=f"{layer_name}__{key}",
        )

    if gdf is not None:
        osm._save_gdf(gdf, stem)
    return gdf, stem


def _load_roads_layer(
    osm: OSMDownloader,
    place: str | None,
    point: tuple[float, float] | None,
    network_type: str,
    use_cache: bool,
    add_elevation: bool,
) -> tuple[gpd.GeoDataFrame | None, gpd.GeoDataFrame | None, str]:
    key = _location_key(place, point)
    stem = osm._gdf_cache_stem(f"roads_{network_type}", key)

    if use_cache:
        nodes, edges = osm._load_roads(stem)
        if nodes is not None and edges is not None:
            return nodes, edges, stem

    if place:
        nodes, edges = osm.roads_from_place(
            place,
            network_type=network_type,
            add_elevation=add_elevation,
            use_cache=False,
            save_cache=False,
        )
    else:
        nodes, edges = osm.roads_from_point(
            point,
            network_type=network_type,
            add_elevation=add_elevation,
            use_cache=False,
            save_cache=False,
            cache_name=key,
        )

    return nodes, edges, stem


def _iter_tiles(bounds: Iterable[float], tile_size: float, overlap: float = 0.0):
    xmin, ymin, xmax, ymax = bounds
    step = max(tile_size - overlap, 1e-6)

    nx = max(1, math.ceil((xmax - xmin) / step))
    ny = max(1, math.ceil((ymax - ymin) / step))

    for ix in range(nx):
        for iy in range(ny):
            x0 = xmin + ix * step
            y0 = ymin + iy * step
            x1 = min(x0 + tile_size, xmax)
            y1 = min(y0 + tile_size, ymax)
            if x1 <= x0 or y1 <= y0:
                continue
            yield ix, iy, box(x0, y0, x1, y1)


def _safe_bounds(*gdfs: gpd.GeoDataFrame | None):
    bounds = [gdf.total_bounds for gdf in gdfs if gdf is not None and not gdf.empty]
    if not bounds:
        return None

    xmin = min(b[0] for b in bounds)
    ymin = min(b[1] for b in bounds)
    xmax = max(b[2] for b in bounds)
    ymax = max(b[3] for b in bounds)
    return xmin, ymin, xmax, ymax


def _clip_gdf_to_tile(gdf: gpd.GeoDataFrame | None, tile_geom):
    if gdf is None or gdf.empty:
        return gdf

    try:
        clipped = gpd.clip(gdf, tile_geom)
        if clipped is not None and not clipped.empty:
            return clipped
    except Exception:
        pass

    try:
        return gdf.loc[gdf.intersects(tile_geom)].copy()
    except Exception:
        return gdf


def _make_builder(terrain, output_dir: Path) -> CityMeshBuilder:
    return CityMeshBuilder(
        terrain_sampler=terrain.height_at, output_dir=str(output_dir)
    )


def _configure_textures(builder: CityMeshBuilder):
    builder.apply_textures(
        building_rules=[],
        road_texture_path=Path(
            "assets/textures/path/cement/Road012A_1K-JPG/Road012A_1K-JPG_Color.jpg"
        ),
        park_texture_path=Path(
            "assets/textures/path/grass/Grass001_1K-JPG/Grass001_1K-JPG_Color.jpg"
        ),
        water_texture_path=Path("assets/textures/water/GPT_muted_green_water.png"),
        sidewalk_texture_path=Path(
            "assets/textures/path/cement/Concrete030_1K-JPG/Concrete030_1K-JPG_Color.jpg"
        ),
    )


def _export_scene(
    terrain,
    output_dir: Path,
    output_name: str,
    buildings: gpd.GeoDataFrame | None,
    roads: gpd.GeoDataFrame | None,
    water: gpd.GeoDataFrame | None,
    parks: gpd.GeoDataFrame | None,
    sidewalks: gpd.GeoDataFrame | None,
    landuse: gpd.GeoDataFrame | None = None,
    parking: gpd.GeoDataFrame | None = None,
    pedestrian: gpd.GeoDataFrame | None = None,
    railway_areas: gpd.GeoDataFrame | None = None,
    terrain_features: gpd.GeoDataFrame | None = None,
):
    builder = _make_builder(terrain, output_dir)
    # Earlier surfaces keep their area; each later surface is clipped to the remainder.
    # builder.add_sidewalks(sidewalks=sidewalks, roads=roads)
    # builder.add_surface_features(pedestrian, "pedestrian")
    # builder.add_surface_features(parking, "parking")
    builder.add_roads(roads)
    # builder.add_water(water)
    # builder.add_parks(parks)
    # builder.add_surface_features(landuse, "landuse")
    # builder.add_surface_features(railway_areas, "railway_areas")
    # builder.add_surface_features(terrain_features, "terrain")
    builder.add_buildings(buildings)
    _configure_textures(builder)

    obj_name = Path(output_name).with_suffix(".obj").name
    bam_name = Path(output_name).with_suffix(".bam").name

    obj_path = builder.export_obj(obj_name)
    bam_path = builder.export_bam(bam_name)
    manifest_path = builder.export_manifest_csv()
    return bam_path, obj_path, manifest_path, builder


def _print_layer_counts(
    buildings: gpd.GeoDataFrame | None,
    nodes: gpd.GeoDataFrame | None,
    roads: gpd.GeoDataFrame | None,
    water: gpd.GeoDataFrame | None,
    parks: gpd.GeoDataFrame | None,
    sidewalks: gpd.GeoDataFrame | None,
    landuse: gpd.GeoDataFrame | None,
    parking: gpd.GeoDataFrame | None,
    pedestrian: gpd.GeoDataFrame | None,
    railway_areas: gpd.GeoDataFrame | None,
    terrain_features: gpd.GeoDataFrame | None,
):
    print(f"Buildings: {0 if buildings is None else len(buildings)}")
    print(f"Road nodes: {0 if nodes is None else len(nodes)}")
    print(f"Road edges: {0 if roads is None else len(roads)}")
    print(f"Water: {0 if water is None else len(water)}")
    print(f"Parks: {0 if parks is None else len(parks)}")
    print(f"Sidewalks: {0 if sidewalks is None else len(sidewalks)}")
    print(f"Landuse surfaces: {0 if landuse is None else len(landuse)}")
    print(f"Parking: {0 if parking is None else len(parking)}")
    print(f"Pedestrian paths/plazas: {0 if pedestrian is None else len(pedestrian)}")
    print(f"Railway areas: {0 if railway_areas is None else len(railway_areas)}")
    print(
        f"Terrain fallback: {0 if terrain_features is None else len(terrain_features)}"
    )


def _load_layers(
    osm: OSMDownloader,
    place: str | None,
    point: tuple[float, float] | None,
    network_type: str,
    use_cache: bool,
    add_elevation: bool,
):
    key = _location_key(place, point)
    landuse_stem = osm._gdf_cache_stem("features", f"landuse__{key}")
    landuse = osm._load_gdf(landuse_stem) if use_cache else None
    if landuse is None:
        if place:
            landuse = osm.features_from_place(
                place,
                tags={"landuse": True},
                add_elevation=False,
                use_cache=False,
                save_cache=True,
                cache_name=f"landuse__{key}",
            )
        else:
            landuse = osm.features_from_point(
                point,
                tags={"landuse": True},
                add_elevation=False,
                use_cache=False,
                save_cache=True,
                cache_name=f"landuse__{key}",
            )

    buildings, bstem = _query_gdf_layer(
        osm, "buildings", place, point, use_cache, add_elevation
    )
    if (
        buildings is not None
        and not buildings.empty
        and landuse is not None
        and not landuse.empty
    ):
        if buildings.crs is not None and landuse.crs != buildings.crs:
            landuse = landuse.to_crs(buildings.crs)
        buildings = osm._attach_landuse_to_buildings(buildings, landuse)
        osm._save_gdf(buildings, bstem)

    nodes, roads, rstem = _load_roads_layer(
        osm, place, point, network_type, use_cache, add_elevation
    )
    water, wstem = _query_gdf_layer(
        osm, "water", place, point, use_cache, add_elevation
    )
    parks, pstem = _query_gdf_layer(
        osm, "parks", place, point, use_cache, add_elevation
    )
    sidewalks, sstem = _query_gdf_layer(
        osm, "sidewalks", place, point, use_cache, add_elevation
    )
    parking, parking_stem = _query_gdf_layer(
        osm, "parking", place, point, use_cache, add_elevation
    )
    pedestrian, pedestrian_stem = _query_gdf_layer(
        osm, "pedestrian", place, point, use_cache, add_elevation
    )
    railway_areas, railway_areas_stem = _query_gdf_layer(
        osm, "railway_areas", place, point, use_cache, add_elevation
    )
    terrain_features, terrain_stem = _query_gdf_layer(
        osm, "terrain", place, point, use_cache, add_elevation
    )
    return (
        (buildings, bstem),
        (nodes, roads, rstem),
        (water, wstem),
        (parks, pstem),
        (sidewalks, sstem),
        (landuse, landuse_stem),
        (parking, parking_stem),
        (pedestrian, pedestrian_stem),
        (railway_areas, railway_areas_stem),
        (terrain_features, terrain_stem),
    )


def main():
    parser = argparse.ArgumentParser(description="Generate a city mesh from OSM data.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--place", type=str, help="OSM place name, e.g. 'Atlanta, Georgia, USA'"
    )
    group.add_argument(
        "--point",
        type=_parse_point,
        help="Latitude,longitude point, e.g. '33.7490,-84.3880'",
    )
    parser.add_argument(
        "--dist",
        type=float,
        default=1000,
        help="Side length of the square OSM query area in meters",
    )
    parser.add_argument(
        "--network_type", type=str, default="drive", help="OSM network type for roads"
    )
    parser.add_argument("--cache_dir", type=str, default="cache")
    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="Load cached GeoDataFrames if available",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Ignore existing cache and re-download"
    )
    parser.add_argument(
        "--save_each",
        action="store_true",
        help="Save each GeoDataFrame as it is fetched",
    )
    parser.add_argument(
        "--output_dir", type=str, default="city_output", help="Output directory"
    )
    parser.add_argument(
        "--output_name", type=str, default="city.glb", help="Output scene filename"
    )
    parser.add_argument(
        "--dem",
        type=str,
        default=None,
        help="Optional DEM raster path for elevation sampling",
    )
    parser.add_argument(
        "--add_elevation",
        action="store_true",
        help="Attach elevation metadata from terrain/DEM",
    )
    parser.add_argument(
        "--tile_size",
        type=float,
        default=None,
        help="Tile size in projected map units. If omitted, export a single full-city scene.",
    )
    parser.add_argument(
        "--tile_overlap",
        type=float,
        default=10.0,
        help="Overlap between adjacent tiles in projected map units.",
    )
    parser.add_argument(
        "--tile_prefix",
        type=str,
        default="tile",
        help="Prefix used when naming tile output folders and files.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Preview the Bam file for verification",
    )
    args = parser.parse_args()

    terrain = RasterTerrainSampler(args.dem) if args.dem else ConstantTerrainSampler()
    osm = OSMDownloader(
        dist=args.dist,
        elevation_sampler=terrain.height_at,
        cache_dir=args.cache_dir,
        overwrite=args.overwrite,
    )
    mb.VERIFY = args.verify
    place = args.place
    point = args.point
    use_cache = args.use_cache and not args.overwrite
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (
        (buildings, bstem),
        (nodes, roads, rstem),
        (water, wstem),
        (parks, pstem),
        (sidewalks, sstem),
        (landuse, landuse_stem),
        (parking, parking_stem),
        (pedestrian, pedestrian_stem),
        (railway_areas, railway_areas_stem),
        (terrain_features, terrain_stem),
    ) = _load_layers(
        osm=osm,
        place=place,
        point=point,
        network_type=args.network_type,
        use_cache=use_cache,
        add_elevation=args.add_elevation,
    )

    _print_layer_counts(
        buildings,
        nodes,
        roads,
        water,
        parks,
        sidewalks,
        landuse,
        parking,
        pedestrian,
        railway_areas,
        terrain_features,
    )

    if args.tile_size is None:
        bam_path, obj_path, manifest_path, builder = _export_scene(
            terrain=terrain,
            output_dir=output_dir,
            output_name=args.output_name,
            buildings=buildings,
            roads=roads,
            water=water,
            parks=parks,
            sidewalks=sidewalks,
            landuse=landuse,
            parking=parking,
            pedestrian=pedestrian,
            railway_areas=railway_areas,
            terrain_features=terrain_features,
        )
        print(f"Exported scene: {bam_path} and {obj_path}")
        print(f"Exported manifest: {manifest_path}")
        print(f"Generated assets: {len(builder.assets)}")
        return

    bounds = _safe_bounds(
        buildings,
        roads,
        water,
        parks,
        sidewalks,
        landuse,
        parking,
        pedestrian,
        railway_areas,
        terrain_features,
    )
    if bounds is None:
        print("No geometry found; nothing to tile.")
        return

    tile_count = 0
    total_assets = 0

    for ix, iy, tile_geom in _iter_tiles(bounds, args.tile_size, args.tile_overlap):
        tile_buildings = _clip_gdf_to_tile(buildings, tile_geom)
        tile_roads = _clip_gdf_to_tile(roads, tile_geom)
        tile_water = _clip_gdf_to_tile(water, tile_geom)
        tile_parks = _clip_gdf_to_tile(parks, tile_geom)
        tile_sidewalks = _clip_gdf_to_tile(sidewalks, tile_geom)
        tile_landuse = _clip_gdf_to_tile(landuse, tile_geom)
        tile_parking = _clip_gdf_to_tile(parking, tile_geom)
        tile_pedestrian = _clip_gdf_to_tile(pedestrian, tile_geom)
        tile_railway_areas = _clip_gdf_to_tile(railway_areas, tile_geom)
        tile_terrain = _clip_gdf_to_tile(terrain_features, tile_geom)

        tile_layers = (
            tile_buildings,
            tile_roads,
            tile_water,
            tile_parks,
            tile_sidewalks,
            tile_landuse,
            tile_parking,
            tile_pedestrian,
            tile_railway_areas,
            tile_terrain,
        )
        if not any(gdf is not None and not gdf.empty for gdf in tile_layers):
            continue

        tile_name = f"{args.tile_prefix}_{ix:03d}_{iy:03d}"
        tile_dir = output_dir / tile_name
        tile_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"{tile_name}: "
            f"buildings={0 if tile_buildings is None else len(tile_buildings)}, "
            f"roads={0 if tile_roads is None else len(tile_roads)}, "
            f"water={0 if tile_water is None else len(tile_water)}, "
            f"parks={0 if tile_parks is None else len(tile_parks)}, "
            f"sidewalks={0 if tile_sidewalks is None else len(tile_sidewalks)}, "
            f"landuse={0 if tile_landuse is None else len(tile_landuse)}, "
            f"parking={0 if tile_parking is None else len(tile_parking)}, "
            f"pedestrian={0 if tile_pedestrian is None else len(tile_pedestrian)}, "
            f"railway_areas={0 if tile_railway_areas is None else len(tile_railway_areas)}, "
            f"terrain={0 if tile_terrain is None else len(tile_terrain)}"
        )

        bam_path, obj_path, manifest_path, builder = _export_scene(
            terrain=terrain,
            output_dir=tile_dir,
            output_name=f"{tile_name}.bam",
            buildings=tile_buildings,
            roads=tile_roads,
            water=tile_water,
            parks=tile_parks,
            sidewalks=tile_sidewalks,
            landuse=tile_landuse,
            parking=tile_parking,
            pedestrian=tile_pedestrian,
            railway_areas=tile_railway_areas,
            terrain_features=tile_terrain,
        )

        print(f"  Exported: {bam_path}")
        print(f"  Exported: {obj_path}")
        print(f"  Exported manifest: {manifest_path}")
        print(f"  Generated assets: {len(builder.assets)}")

        tile_count += 1
        total_assets += len(builder.assets)

    print(f"Finished exporting {tile_count} tiles.")
    print(f"Total generated assets: {total_assets}")


if __name__ == "__main__":
    main()

    """
    Examples to run:
    1. Full city:
       python main.py --place "Baltimore, Maryland, USA" --output_dir assets/Terrain/Generate/baltimore --output_name baltimore.bam --add_elevation --use_cache

    2. Tiled export:
       python main.py --place "Baltimore, Maryland, USA" --output_dir assets/Terrain/Generate/baltimore --output_name baltimore.bam --add_elevation --use_cache --dist 1000.0 --verify

    3. Point-based location:
       python main.py --point 39.358575,-76.682986 --dist 200 --output_dir assets/Terrain/Generate/baltimore_neighborhood --output_name baltimore_neighborhood.bam --add_elevation --use_cache

    4. With DEM data:
       python main.py --place "Atlanta, Georgia, USA" --dem atlanta_dem.tif --output_dir assets/Terrain/Generate/baltimore --output_name baltimore.bam --add_elevation
    """
