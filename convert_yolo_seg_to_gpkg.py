#!/usr/bin/env python3
"""
Convert YOLO-format segmentation labels (normalized polygons) to geospatial polygons (GPKG).

- Reads georeferencing from companion .jpg.aux.xml using rasterio/GDAL
- Supports very large datasets via chunked append writes
- Adds image path and class_id fields
"""

from pathlib import Path
from typing import List, Tuple, Optional
import sys

import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.errors import TopologicalError
import rasterio
from rasterio.transform import Affine
from tqdm import tqdm


def _find_image_for_label(stem: str, images_root: Path) -> Optional[Path]:
    """Find the image file by stem under images_root (tries common JPEG extensions)."""
    for ext in (".jpg", ".jpeg", ".JPG", ".JPEG"):
        p = images_root / f"{stem}{ext}"
        if p.exists():
            return p
    # fallback: search (slower)
    candidates = list(images_root.rglob(f"{stem}.jpg")) + list(images_root.rglob(f"{stem}.jpeg"))
    return candidates[0] if candidates else None


def _parse_label_line(line: str) -> Tuple[int, List[Tuple[float, float]]]:
    """
    Parse one YOLO-seg line: 'cls x1 y1 x2 y2 ...' (normalized coords).
    Returns (class_id, [(xn, yn), ...])
    """
    parts = line.strip().split()
    if len(parts) < 6 or len(parts) % 2 == 0:
        raise ValueError(f"Bad YOLO-seg line (need class + x y pairs): {line[:120]} ...")
    cls = int(float(parts[0]))
    nums = list(map(float, parts[1:]))
    pts = [(nums[i], nums[i + 1]) for i in range(0, len(nums), 2)]
    return cls, pts


def _img_norm_to_pixel(pts_norm: List[Tuple[float, float]], width: int, height: int) -> np.ndarray:
    """
    Convert normalized image coords (0..1) to pixel-space (col=x, row=y).
    YOLO normalized origin is top-left; raster rows increase downward, which matches.
    """
    arr = np.asarray(pts_norm, dtype=np.float64)
    # Multiply by image width/height
    px = np.empty_like(arr)
    px[:, 0] = arr[:, 0] * width   # x → columns
    px[:, 1] = arr[:, 1] * height  # y → rows
    return px


def _pixel_to_crs(px: np.ndarray, transform: Affine) -> np.ndarray:
    """
    Map pixel coordinates (col, row) to CRS coordinates using the dataset Affine transform.
    In GDAL/rasterio convention, Affine maps (col, row) to the upper-left corner of that pixel.
    """
    # Vectorized affine: x = a*col + b*row + c ; y = d*col + e*row + f
    a, b, c, d, e, f = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
    xs = a * px[:, 0] + b * px[:, 1] + c
    ys = d * px[:, 0] + e * px[:, 1] + f
    return np.column_stack([xs, ys])


def convert_yolo_seg_to_gpkg(
    labels_dir: Path,
    images_root: Path,
    out_gpkg: Path,
    layer: str = "yolo_seg",
    chunk_size: int = 5000,
    overwrite: bool = True
):
    labels_dir = labels_dir.resolve()
    images_root = images_root.resolve()
    out_gpkg = out_gpkg.resolve()

    if overwrite and out_gpkg.exists():
        out_gpkg.unlink()

    label_files = sorted(labels_dir.glob("*.txt"))
    if not label_files:
        print(f"No label files found in {labels_dir}", file=sys.stderr)
        return

    wrote_any = False
    batch_records = []
    batch_geoms = []

    pbar = tqdm(label_files, desc="Converting YOLO-seg to polygons", unit="img")

    for lbl_path in pbar:
        stem = lbl_path.stem
        img_path = _find_image_for_label(stem, images_root)
        if img_path is None:
            # Skip if the matching image is missing
            continue

        # Open raster to fetch transform & crs and width/height
        try:
            with rasterio.open(img_path) as ds:
                width, height = ds.width, ds.height
                transform = ds.transform
                crs = ds.crs
        except Exception as e:
            print(f"⚠️ Could not open raster for {img_path}: {e}", file=sys.stderr)
            continue

        # Parse each polygon line in the label file
        try:
            lines = [ln for ln in lbl_path.read_text().splitlines() if ln.strip()]
        except UnicodeDecodeError:
            # If encoding issues, try latin-1 fallback
            with open(lbl_path, "r", encoding="latin-1") as f:
                lines = [ln for ln in f.read().splitlines() if ln.strip()]

        for line in lines:
            try:
                cls, pts_norm = _parse_label_line(line)
            except ValueError as ve:
                print(f"⚠️ Skipping bad line in {lbl_path.name}: {ve}", file=sys.stderr)
                continue

            # Norm → pixel → CRS
            px = _img_norm_to_pixel(pts_norm, width, height)
            xy = _pixel_to_crs(px, transform)

            # Ensure closed ring
            if not (np.isclose(xy[0], xy[-1]).all()):
                xy = np.vstack([xy, xy[0]])

            # Build polygon and attempt light fix if invalid
            try:
                poly = Polygon(xy)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                # Skip empty/invalid after fix
                if (poly is None) or poly.is_empty:
                    continue
            except TopologicalError:
                continue

            batch_records.append({"image": str(img_path), "class_id": int(cls)})
            batch_geoms.append(poly)

            # Write in chunks to keep memory steady
            if len(batch_geoms) >= chunk_size:
                gdf = gpd.GeoDataFrame(batch_records, geometry=batch_geoms, crs=crs)
                gdf.to_file(out_gpkg, layer=layer, driver="GPKG", mode="w" if not wrote_any else "a")
                wrote_any = True
                batch_records.clear()
                batch_geoms.clear()

    # Flush remainder
    if batch_geoms:
        # Use last seen CRS if available; if not, default to EPSG:2193 as per your data
        crs_out = crs if 'crs' in locals() and crs else "EPSG:2193"
        gdf = gpd.GeoDataFrame(batch_records, geometry=batch_geoms, crs=crs_out)
        gdf.to_file(out_gpkg, layer=layer, driver="GPKG", mode="w" if not wrote_any else "a")
        wrote_any = True

    if wrote_any:
        print(f"✅ Wrote polygons to {out_gpkg} (layer='{layer}')")
    else:
        print("No polygons were written (check inputs).")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert YOLO segmentation polygons to georeferenced GeoPackage'
    )
    parser.add_argument('--labels', type=str, required=True,
                        help='Directory containing YOLO segmentation labels')
    parser.add_argument('--images', type=str, required=True,
                        help='Root directory containing images')
    parser.add_argument('--output', type=str, required=True,
                        help='Output GeoPackage file path')
    parser.add_argument('--layer', type=str, default='polygons',
                        help='Output layer name (default: polygons)')
    parser.add_argument('--chunk-size', type=int, default=5000,
                        help='Chunk size for memory-efficient writes (default: 5000)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite output file if it exists')

    args = parser.parse_args()

    convert_yolo_seg_to_gpkg(
        labels_dir=Path(args.labels),
        images_root=Path(args.images),
        out_gpkg=Path(args.output),
        layer=args.layer,
        chunk_size=args.chunk_size,
        overwrite=args.overwrite
    )

