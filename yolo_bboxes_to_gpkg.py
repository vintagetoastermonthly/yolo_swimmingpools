#!/usr/bin/env python3
"""
Convert YOLO detection bboxes (cx cy w h normalized) to geospatial rectangle polygons (GPKG).

- Reads georeferencing from .jpg.aux.xml via rasterio
- Supports optional confidence per line
- Adds class_id, class_name, confidence, and image path
- Handles large sets with chunked writes and a progress bar
"""

from pathlib import Path
from typing import Optional, Tuple, List
import sys

import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
import rasterio
from rasterio.transform import Affine
from tqdm import tqdm

# --- CONFIG: update these paths for your environment ---
LABELS_DIR = Path(r"/mnt/d/Auckland/swimmingpools/output_water/labels")
IMAGES_ROOT = Path(r"/mnt/d/Auckland/water_chips_0075m")
OUT_GPKG = Path(r"/mnt/d/Auckland/swimmingpools/water2017_bboxes.gpkg")
LAYER_NAME = "pools_bbox"
# ------------------------------------------------------

CLASS_MAP = {
    0: "Built Pools",
    1: "Temporary Pools",
}
IMG_EXTS = (".jpg", ".jpeg", ".JPG", ".JPEG")

def find_image_for_label(stem: str, images_root: Path) -> Optional[Path]:
    """Find matching image by stem under images_root."""
    # fast direct checks
    for ext in IMG_EXTS:
        p = images_root / f"{stem}{ext}"
        if p.exists():
            return p
    # fallback recursive search (stop at first hit)
    for ext in IMG_EXTS:
        for p in images_root.rglob(f"{stem}{ext}"):
            if p.exists():
                return p
    return None

def parse_yolo_det_line(line: str) -> Tuple[int, float, float, float, float, Optional[float]]:
    """
    Parse a YOLO detection line:
      class cx cy w h [conf]
    Returns: (class_id, cx, cy, w, h, conf_or_None)
    """
    parts = line.strip().split()
    if len(parts) < 5:
        raise ValueError(f"Bad line (need at least 5 tokens): {line[:160]} ...")
    cls = int(float(parts[0]))
    cx = float(parts[1]); cy = float(parts[2]); w = float(parts[3]); h = float(parts[4])
    conf = float(parts[5]) if len(parts) >= 6 else None
    return cls, cx, cy, w, h, conf

def bbox_norm_to_pixel(cx: float, cy: float, w: float, h: float, width: int, height: int) -> np.ndarray:
    """
    Convert normalized bbox (cx,cy,w,h) to pixel XY corners (x1,y1,x2,y2) in image space.
    """
    cx_px = cx * width
    cy_px = cy * height
    w_px = w * width
    h_px = h * height
    x1 = cx_px - w_px / 2.0
    y1 = cy_px - h_px / 2.0
    x2 = cx_px + w_px / 2.0
    y2 = cy_px + h_px / 2.0
    return np.array([x1, y1, x2, y2], dtype=np.float64)

def rect_to_polygon_xyxy(x1: float, y1: float, x2: float, y2: float, transform: Affine) -> Polygon:
    """
    Convert pixel-space rectangle to CRS polygon using the raster affine transform.
    Pixel vertices are mapped to CRS via the affine (upper-left pixel convention).
    """
    # rectangle corners in pixel coords (col, row)
    px_pts = np.array([
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2],
        [x1, y1]
    ], dtype=np.float64)

    # affine to CRS (vectorized)
    a, b, c, d, e, f = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
    xs = a * px_pts[:, 0] + b * px_pts[:, 1] + c
    ys = d * px_pts[:, 0] + e * px_pts[:, 1] + f

    return Polygon(np.column_stack([xs, ys]))

def main():
    labels_dir = LABELS_DIR.resolve()
    images_root = IMAGES_ROOT.resolve()
    out_gpkg = OUT_GPKG.resolve()

    # sanity
    if not labels_dir.exists():
        print(f"❌ Labels dir not found: {labels_dir}", file=sys.stderr)
        return
    if not images_root.exists():
        print(f"❌ Images root not found: {images_root}", file=sys.stderr)
        return
    if out_gpkg.exists():
        out_gpkg.unlink()

    label_files = sorted(labels_dir.rglob("*.txt"))
    if not label_files:
        print(f"❌ No label files found under {labels_dir}", file=sys.stderr)
        return

    print(f"Labels dir:  {labels_dir}")
    print(f"Images root: {images_root}")
    print(f"Output GPKG: {out_gpkg} (layer='{LAYER_NAME}')")
    print(f"Found {len(label_files)} label files\n")

    batch_records: List[dict] = []
    batch_geoms: List[Polygon] = []
    wrote_any = False
    chunk_size = 10000  # tune for memory vs write frequency

    n_files = 0
    n_images_found = 0
    n_lines = 0
    n_written = 0
    n_missing_img = 0
    n_bad_lines = 0

    last_crs = None

    for lbl_path in tqdm(label_files, desc="Converting YOLO bboxes → polygons", unit="file"):
        n_files += 1
        stem = lbl_path.stem
        img_path = find_image_for_label(stem, images_root)
        if img_path is None:
            n_missing_img += 1
            if n_missing_img <= 5:
                print(f"⚠️ Missing image for label {lbl_path.name}", file=sys.stderr)
            continue
        n_images_found += 1

        # Open raster for geotransform
        try:
            with rasterio.open(img_path) as ds:
                width, height = ds.width, ds.height
                transform = ds.transform
                crs = ds.crs
                last_crs = crs
        except Exception as e:
            print(f"⚠️ Could not open raster {img_path}: {e}", file=sys.stderr)
            continue

        # Read lines
        try:
            lines = [ln for ln in lbl_path.read_text().splitlines() if ln.strip()]
        except UnicodeDecodeError:
            with open(lbl_path, "r", encoding="latin-1") as f:
                lines = [ln for ln in f.read().splitlines() if ln.strip()]

        # Parse and convert each bbox
        for line in lines:
            n_lines += 1
            try:
                cls, cx, cy, w, h, conf = parse_yolo_det_line(line)
            except ValueError as ve:
                n_bad_lines += 1
                if n_bad_lines <= 5:
                    print(f"⚠️ Bad line in {lbl_path.name}: {ve}", file=sys.stderr)
                continue

            # Norm → pixel xyxy
            x1, y1, x2, y2 = bbox_norm_to_pixel(cx, cy, w, h, width, height)
            # Build CRS polygon
            poly = rect_to_polygon_xyxy(x1, y1, x2, y2, transform)

            # Attributes
            rec = {
                "image": str(img_path),
                "class_id": int(cls),
                "class_name": CLASS_MAP.get(int(cls), str(int(cls))),
                "confidence": float(conf) if conf is not None else None,
            }

            batch_records.append(rec)
            batch_geoms.append(poly)
            n_written += 1

            # Chunked write
            if len(batch_geoms) >= chunk_size:
                gdf = gpd.GeoDataFrame(batch_records, geometry=batch_geoms, crs=crs)
                gdf.to_file(out_gpkg, layer=LAYER_NAME, driver="GPKG", mode="w" if not wrote_any else "a")
                wrote_any = True
                batch_records.clear()
                batch_geoms.clear()

    # flush remainder
    if batch_geoms:
        crs_out = last_crs if last_crs else "EPSG:2193"
        gdf = gpd.GeoDataFrame(batch_records, geometry=batch_geoms, crs=crs_out)
        gdf.to_file(out_gpkg, layer=LAYER_NAME, driver="GPKG", mode="w" if not wrote_any else "a")
        wrote_any = True

    print("\n--- Summary ---")
    print(f"Label files processed : {n_files}")
    print(f"Images matched        : {n_images_found}")
    print(f"Label lines read      : {n_lines}")
    print(f"Polygons written      : {n_written}")
    print(f"Missing images        : {n_missing_img}")
    print(f"Bad label lines       : {n_bad_lines}")

    if wrote_any:
        print(f"\n✅ Wrote bbox polygons to {out_gpkg} (layer='{LAYER_NAME}')")
    else:
        print("\n❌ No features were written — check paths and label formats.")

if __name__ == "__main__":
    main()

