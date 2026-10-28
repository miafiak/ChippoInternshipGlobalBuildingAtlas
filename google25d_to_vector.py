"""
Convert Google Open Buildings 2.5D Temporal (raster) into building polygons (vector).

Steps
  1. Download building_presence / building_height / building_fractional_count for an AOI
     and a year from Earth Engine, in UTM (metric, square 4 m pixels), in tiles.
  2. Threshold the presence band to get a building mask.
  3. Optionally split touching buildings with a distance-transform watershed.
  4. Polygonize, attach per-polygon height / area / count statistics, simplify, and export.

Examples
  python google25d_to_vector.py --bbox 24.80 13.45 24.90 13.55 --year 2019 --out buildings.parquet
  python google25d_to_vector.py --aoi camp.shp --out buildings.geojson --split-min-distance 3

Notes
  - Requires an Earth Engine account and project: `earthengine authenticate` once, then pass --project.
  - Google 2.5D only covers Africa, South/Southeast Asia and Latin America & the Caribbean.
  - The merged raster is held in memory, so use this for city-sized AOIs; run larger regions in pieces.
  - The output is a modeled approximation of footprints, not surveyed footprints.
"""
import argparse
import math
import os
import tempfile
import time
import urllib.request

import ee
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize, shapes
from rasterio.merge import merge
from scipy import ndimage as ndi
from shapely import force_2d
from shapely.geometry import box, shape
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

COLLECTION = "GOOGLE/Research/open-buildings-temporal/v1"
BANDS = ["building_presence", "building_height", "building_fractional_count"]
PIXEL_M = 4  # native resolution of the product
VECTOR_COLUMNS = [
    "height_mean", "height_median", "height_max", "presence_mean", "fractional_count_sum", "area_m2",
]


def load_aoi(bbox, aoi_path):
    """Return the AOI as a single 2D shapely geometry in EPSG:4326."""
    if bbox:
        return box(*bbox)
    gdf = gpd.read_file(aoi_path)
    if gdf.crs is None:
        raise ValueError("AOI file has no CRS defined; check the .prj file.")
    return force_2d(gdf.to_crs("EPSG:4326").geometry.union_all())


def utm_crs_for(geom):
    """EPSG code string of the UTM zone containing the geometry's centroid."""
    c = geom.centroid
    zone = int((c.x + 180) // 6) + 1
    return f"EPSG:{(32600 if c.y >= 0 else 32700) + zone}"


def build_image(aoi_wgs84, year):
    """Mosaic of the 2.5D bands for the year, as in the notebook's get_2_5d_stats()."""
    ee_geom = ee.Geometry(aoi_wgs84.__geo_interface__)
    collection = (
        ee.ImageCollection(COLLECTION)
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .filterBounds(ee_geom)
    )
    n = collection.size().getInfo()
    if n == 0:
        raise RuntimeError(
            f"No 2.5D images for {year} in this AOI. It is probably outside the coverage "
            "(Africa, South/Southeast Asia, Latin America & the Caribbean) or the year is unavailable."
        )
    print(f"Found {n} source image(s) for {year}.")
    # unmask(0) so empty pixels become 0 instead of nodata
    return collection.mosaic().select(BANDS).toFloat().unmask(0)


def download_tiles(image, aoi_utm, crs, tile_m, out_dir, retries=4):
    """Download the image as GeoTIFF tiles of tile_m x tile_m metres covering the AOI."""
    minx, miny, maxx, maxy = aoi_utm.bounds
    xs = np.arange(math.floor(minx / PIXEL_M) * PIXEL_M, maxx, tile_m)
    ys = np.arange(math.floor(miny / PIXEL_M) * PIXEL_M, maxy, tile_m)
    paths = []
    todo = [(x, y) for x in xs for y in ys if box(x, y, x + tile_m, y + tile_m).intersects(aoi_utm)]
    for i, (x, y) in enumerate(todo, 1):
        path = os.path.join(out_dir, f"tile_{i:04d}.tif")
        region = ee.Geometry.Rectangle([x, y, x + tile_m, y + tile_m], proj=crs, geodesic=False)
        for attempt in range(1, retries + 1):
            try:
                url = image.getDownloadURL(
                    {"region": region, "scale": PIXEL_M, "crs": crs, "format": "GEO_TIFF"}
                )
                urllib.request.urlretrieve(url, path)
                break
            except Exception as exc:  # EE and network errors are varied; retry then fail loudly
                if attempt == retries:
                    raise RuntimeError(f"Tile {i}/{len(todo)} failed after {retries} attempts: {exc}")
                time.sleep(2 * attempt)
        print(f"Downloaded tile {i}/{len(todo)}")
        paths.append(path)
    return paths


def merge_tiles(paths):
    """Merge tile GeoTIFFs into (bands, H, W) array + affine transform."""
    datasets = [rasterio.open(p) for p in paths]
    try:
        arr, transform = merge(datasets)
    finally:
        for ds in datasets:
            ds.close()
    return arr.astype("float32"), transform


def split_blobs(mask, sigma, min_distance):
    """Label building pixels; split touching buildings with a distance-transform watershed.

    Components that yield no peak keep a single label of their own.
    """
    components, n_comp = ndi.label(mask)
    if min_distance <= 0:
        return components
    dist = ndi.gaussian_filter(ndi.distance_transform_edt(mask), sigma)
    peaks = peak_local_max(dist, min_distance=min_distance, labels=components, exclude_border=False)
    markers = np.zeros(mask.shape, dtype="int32")
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    labels = watershed(-dist, markers, mask=mask)
    missed = mask & (labels == 0)  # components with no peak
    labels[missed] = components[missed] + labels.max()
    return labels


def vectorize(arr, transform, crs, aoi_utm, threshold, sigma, min_distance, simplify_m, min_area_m2):
    presence, height, frac_count = arr
    aoi_mask = rasterize([(aoi_utm, 1)], out_shape=presence.shape, transform=transform, fill=0).astype(bool)
    mask = (presence > threshold) & aoi_mask
    if not mask.any():
        return gpd.GeoDataFrame(
            {c: [] for c in VECTOR_COLUMNS}, geometry=gpd.GeoSeries([], crs=crs), crs=crs
        )

    labels = split_blobs(mask, sigma, min_distance)
    idx = np.unique(labels[labels > 0])

    stats = {
        "height_mean": ndi.mean(height, labels, idx),
        "height_median": ndi.median(height, labels, idx),
        "height_max": ndi.maximum(height, labels, idx),
        "presence_mean": ndi.mean(presence, labels, idx),
        # Sum of the building_fractional_count band. Its scaling is unverified, so do NOT read it as a building count.
        "fractional_count_sum": ndi.sum(frac_count, labels, idx),
    }
    stats_gdf = gpd.GeoDataFrame(stats, index=idx)

    polys = [
        {"label": int(v), "geometry": shape(g)}
        for g, v in shapes(labels.astype("int32"), mask=labels > 0, transform=transform)
    ]
    gdf = gpd.GeoDataFrame(polys, crs=crs).dissolve(by="label").join(stats_gdf)  # one row per label
    gdf["area_m2"] = gdf.geometry.area
    gdf = gdf[gdf["area_m2"] >= min_area_m2].copy()
    if simplify_m > 0:
        gdf["geometry"] = gdf.geometry.simplify(simplify_m, preserve_topology=True)
    return gdf.reset_index(drop=True)


def fetch_raster(aoi_wgs84, year=2019, tile_m=4000):
    """Download the 2.5D bands for the AOI once. Pass the result to vectorize_raster() as often as needed."""
    crs = utm_crs_for(aoi_wgs84)
    aoi_utm = gpd.GeoSeries([aoi_wgs84], crs="EPSG:4326").to_crs(crs).iloc[0]
    print(f"AOI area: {aoi_utm.area / 1e6:.2f} km2, working CRS {crs}")
    image = build_image(aoi_wgs84, year)
    with tempfile.TemporaryDirectory() as tmp:
        paths = download_tiles(image, aoi_utm, crs, tile_m, tmp)
        arr, transform = merge_tiles(paths)
    return {"arr": arr, "transform": transform, "crs": crs, "aoi_utm": aoi_utm, "year": year}


def vectorize_raster(raster, threshold=0.5, split_sigma=1.0, split_min_distance=3, simplify_m=1.5, min_area_m2=20.0):
    """Polygons (EPSG:4326) from a raster returned by fetch_raster(). Empty GeoDataFrame if nothing passes."""
    gdf = vectorize(raster["arr"], raster["transform"], raster["crs"], raster["aoi_utm"],
                    threshold, split_sigma, split_min_distance, simplify_m, min_area_m2)
    gdf["year"] = raster["year"]
    gdf["source"] = "Google Open Buildings 2.5D Temporal (vectorized)"
    gdf.insert(0, "id", np.arange(len(gdf)))
    return gdf.to_crs("EPSG:4326")


def google25d_vectors(aoi_wgs84, year=2019, tile_m=4000, **vector_kwargs):
    """One call: download + vectorize. vector_kwargs go to vectorize_raster()."""
    return vectorize_raster(fetch_raster(aoi_wgs84, year, tile_m), **vector_kwargs)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--bbox", nargs=4, type=float, metavar=("MINLON", "MINLAT", "MAXLON", "MAXLAT"))
    src.add_argument("--aoi", help="Shapefile / GeoJSON / GeoPackage with the AOI")
    p.add_argument("--year", type=int, default=2019, help="Default 2019 (matches GBA)")
    p.add_argument("--out", required=True, help=".parquet / .geoparquet for GeoParquet, anything else for GeoJSON")
    p.add_argument("--project", help="Google Cloud project ID for Earth Engine")
    p.add_argument("--threshold", type=float, default=0.5, help="building_presence cutoff (default 0.5)")
    p.add_argument("--split-min-distance", type=int, default=3,
                   help="Min pixel distance between building centres for blob splitting; 0 disables (default 3)")
    p.add_argument("--split-sigma", type=float, default=1.0, help="Gaussian smoothing before peak finding (default 1.0)")
    p.add_argument("--simplify", type=float, default=1.5, help="Simplification tolerance in metres, 0 disables")
    p.add_argument("--min-area", type=float, default=20.0, help="Drop polygons smaller than this many m2")
    p.add_argument("--tile-size", type=int, default=4000, help="Download tile edge in metres (default 4000)")
    args = p.parse_args()

    ee.Initialize(project=args.project) if args.project else ee.Initialize()

    aoi = load_aoi(args.bbox, args.aoi)
    try:
        gdf = google25d_vectors(
            aoi, args.year, args.tile_size, threshold=args.threshold, split_sigma=args.split_sigma,
            split_min_distance=args.split_min_distance, simplify_m=args.simplify, min_area_m2=args.min_area,
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc))
    if gdf.empty:
        raise SystemExit("No pixels above the presence threshold in this AOI.")

    if args.out.lower().endswith((".parquet", ".geoparquet")):
        gdf.to_parquet(args.out)
    else:
        gdf.to_file(args.out, driver="GeoJSON")
    print(f"Wrote {len(gdf)} polygons to {args.out}")


if __name__ == "__main__":
    main()
