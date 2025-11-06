#!/usr/bin/env python3
"""
Visualize YOLO segmentation outputs overlaid on images.

Usage:
  python visualize_segmentations.py --images IMAGES_DIR \
                                     --labels LABELS_DIR \
                                     --outdir OUTPUT_DIR \
                                     --sample N
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple
import random

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm


def parse_yolo_segmentation(line: str) -> Tuple[int, List[Tuple[float, float]]]:
    """
    Parse YOLO segmentation format: class_id x1 y1 x2 y2 x3 y3 ...

    Returns:
        (class_id, list of (x, y) normalized coordinates)
    """
    parts = line.strip().split()
    if len(parts) < 7:  # class_id + at least 3 vertices (6 coords)
        raise ValueError(f"Invalid YOLO segmentation line: {line}")

    class_id = int(parts[0])

    # Parse vertices
    coords = list(map(float, parts[1:]))
    if len(coords) % 2 != 0:
        raise ValueError(f"Odd number of coordinates: {line}")

    vertices = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]

    return class_id, vertices


def denormalize_polygon(vertices: List[Tuple[float, float]], img_w: int, img_h: int) -> np.ndarray:
    """
    Convert normalized YOLO vertices to pixel coordinates.

    Returns:
        np.ndarray of shape (N, 1, 2) for cv2.polylines/fillPoly
    """
    pixel_vertices = []
    for x_norm, y_norm in vertices:
        x_px = int(x_norm * img_w)
        y_px = int(y_norm * img_h)
        # Clip to image bounds
        x_px = max(0, min(x_px, img_w - 1))
        y_px = max(0, min(y_px, img_h - 1))
        pixel_vertices.append([x_px, y_px])

    return np.array(pixel_vertices, dtype=np.int32).reshape((-1, 1, 2))


def visualize_segmentation(
    image_path: Path,
    label_path: Path,
    output_path: Path,
    class_names: List[str],
    colors: List[Tuple[int, int, int]],
    alpha: float = 0.4
):
    """
    Visualize segmentation polygons on image.

    Args:
        image_path: Path to image
        label_path: Path to YOLO segmentation label file
        output_path: Path to save annotated image
        class_names: List of class names
        colors: List of BGR colors for each class
        alpha: Transparency for filled polygons (0=transparent, 1=opaque)
    """
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"⚠️  Failed to load image: {image_path}")
        return

    img_h, img_w = img.shape[:2]

    # Create overlay for semi-transparent polygons
    overlay = img.copy()

    # Read label file
    if not label_path.exists():
        print(f"⚠️  Label file not found: {label_path}")
        return

    with open(label_path, 'r') as f:
        lines = f.readlines()

    if not lines:
        print(f"⚠️  Empty label file: {label_path}")
        return

    # Parse and draw each segmentation
    num_polygons = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue

        try:
            class_id, vertices = parse_yolo_segmentation(line)

            # Get color for this class
            color = colors[class_id % len(colors)]

            # Convert to pixel coordinates
            polygon = denormalize_polygon(vertices, img_w, img_h)

            # Draw filled polygon on overlay
            cv2.fillPoly(overlay, [polygon], color)

            # Draw polygon outline on original image
            cv2.polylines(img, [polygon], isClosed=True, color=color, thickness=2)

            # Draw class label
            centroid = polygon.mean(axis=0)[0].astype(int)
            class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
            cv2.putText(
                img,
                class_name,
                tuple(centroid),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
            cv2.putText(
                img,
                class_name,
                tuple(centroid),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA
            )

            num_polygons += 1

        except ValueError as e:
            print(f"⚠️  Skipping invalid line in {label_path}: {e}")
            continue

    # Blend overlay with original image
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # Save annotated image
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)

    return num_polygons


def find_image_label_pairs(images_dir: Path, labels_dir: Path) -> List[Tuple[Path, Path]]:
    """
    Find all images with corresponding label files.
    """
    image_extensions = {'.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG'}
    pairs = []

    for img_path in images_dir.rglob('*'):
        if img_path.suffix in image_extensions:
            # Look for corresponding label
            label_path = labels_dir / img_path.relative_to(images_dir).with_suffix('.txt')

            if label_path.exists():
                pairs.append((img_path, label_path))

    return pairs


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize YOLO segmentation outputs'
    )
    parser.add_argument('--images', type=str, required=True,
                        help='Directory containing images')
    parser.add_argument('--labels', type=str, required=True,
                        help='Directory containing YOLO segmentation labels')
    parser.add_argument('--outdir', type=str, required=True,
                        help='Output directory for annotated images')
    parser.add_argument('--sample', type=int, default=None,
                        help='Visualize only N random samples (default: all)')
    parser.add_argument('--alpha', type=float, default=0.4,
                        help='Transparency for filled polygons (default: 0.4)')
    parser.add_argument('--class-names', type=str, nargs='+',
                        default=['SwimmingPool', 'PaddlingPool'],
                        help='Class names (default: SwimmingPool PaddlingPool)')

    return parser.parse_args()


def main():
    args = parse_args()

    # Convert paths
    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    outdir = Path(args.outdir)

    # Validate inputs
    if not images_dir.exists():
        print(f"❌ Images directory not found: {images_dir}")
        sys.exit(1)

    if not labels_dir.exists():
        print(f"❌ Labels directory not found: {labels_dir}")
        sys.exit(1)

    # Create output directory
    outdir.mkdir(parents=True, exist_ok=True)

    # Define colors for classes (BGR format)
    colors = [
        (0, 255, 255),    # Yellow for class 0 (SwimmingPool)
        (255, 0, 255),    # Magenta for class 1 (PaddlingPool)
        (0, 255, 0),      # Green
        (255, 0, 0),      # Blue
        (0, 0, 255),      # Red
    ]

    print(f"\n🎨 Visualizing segmentations")
    print(f"   Images:      {images_dir}")
    print(f"   Labels:      {labels_dir}")
    print(f"   Output:      {outdir}")
    print(f"   Alpha:       {args.alpha}")
    print(f"   Class names: {args.class_names}")

    # Find image-label pairs
    print(f"\n🔍 Finding images with segmentation labels...")
    pairs = find_image_label_pairs(images_dir, labels_dir)

    if not pairs:
        print(f"❌ No images with corresponding labels found!")
        sys.exit(1)

    print(f"✅ Found {len(pairs)} images with labels")

    # Sample if requested
    if args.sample is not None and args.sample < len(pairs):
        print(f"🎲 Randomly sampling {args.sample} images...")
        pairs = random.sample(pairs, args.sample)

    # Visualize
    print(f"\n🖼️  Visualizing {len(pairs)} images...")
    total_polygons = 0

    for img_path, label_path in tqdm(pairs, desc="Visualizing"):
        # Determine output path
        rel_path = img_path.relative_to(images_dir)
        output_path = outdir / rel_path

        # Visualize
        num_polygons = visualize_segmentation(
            img_path,
            label_path,
            output_path,
            args.class_names,
            colors,
            args.alpha
        )

        if num_polygons:
            total_polygons += num_polygons

    # Summary
    print(f"\n✅ Visualization complete!")
    print(f"   Images processed:  {len(pairs)}")
    print(f"   Total polygons:    {total_polygons}")
    print(f"   Output directory:  {outdir}")


if __name__ == '__main__':
    main()
