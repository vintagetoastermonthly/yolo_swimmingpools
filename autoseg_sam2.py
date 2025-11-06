#!/usr/bin/env python3
"""
Auto-segmentation using RT-DETR detections + SAM (Segment Anything Model).

This script uses HuggingFace Transformers SAM implementation (Apache 2.0 licensed).
It takes bounding box detections from RT-DETR and uses them as prompts for SAM
to generate accurate segmentation polygons.

Workflow:
  1. Run RT-DETR inference to get bounding boxes: infer_hf.py --save-txt
  2. Run this script to convert bboxes → segmentation polygons via SAM
  3. Convert polygons to georeferenced GPKG: convert_yolo_seg_to_gpkg.py

Usage:
  python autoseg_sam2.py --images images/ \
                         --labels labels/ \
                         --outdir labels_seg/ \
                         --sam-model facebook/sam-vit-base

  # With confidence filtering and larger model
  python autoseg_sam2.py --images images/ \
                         --labels labels/ \
                         --outdir labels_seg/ \
                         --conf 0.5 \
                         --sam-model facebook/sam-vit-large
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import warnings

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import SamModel, Sam2Model, Sam2Processor, AutoProcessor
import cv2


def parse_yolo_detection(line: str) -> Tuple[int, float, float, float, float, float]:
    """
    Parse YOLO detection format: class cx cy w h [conf]
    Returns: (class_id, cx, cy, w, h, confidence)
    """
    parts = line.strip().split()
    if len(parts) < 5:
        raise ValueError(f"Invalid YOLO detection line: {line}")

    class_id = int(parts[0])
    cx, cy, w, h = map(float, parts[1:5])
    conf = float(parts[5]) if len(parts) >= 6 else 1.0

    return class_id, cx, cy, w, h, conf


def yolo_to_xyxy(cx: float, cy: float, w: float, h: float, img_w: int, img_h: int) -> List[int]:
    """
    Convert YOLO normalized bbox (cx, cy, w, h) to pixel coordinates [x1, y1, x2, y2]
    """
    # Denormalize
    cx_px = cx * img_w
    cy_px = cy * img_h
    w_px = w * img_w
    h_px = h * img_h

    # Convert to corners
    x1 = int(cx_px - w_px / 2)
    y1 = int(cy_px - h_px / 2)
    x2 = int(cx_px + w_px / 2)
    y2 = int(cy_px + h_px / 2)

    # Clip to image bounds
    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w - 1))
    y2 = max(0, min(y2, img_h - 1))

    return [x1, y1, x2, y2]


def mask_to_polygon(mask: np.ndarray, simplify_tolerance: float = 1.0) -> Optional[List[Tuple[float, float]]]:
    """
    Convert binary mask to polygon vertices using OpenCV contour detection.

    Args:
        mask: Binary mask (H, W) with True/1 for object pixels
        simplify_tolerance: Douglas-Peucker simplification tolerance (pixels)

    Returns:
        List of (x, y) vertices in pixel coordinates, or None if no valid contour
    """
    # Ensure uint8
    mask_uint8 = (mask * 255).astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Take largest contour
    contour = max(contours, key=cv2.contourArea)

    # Simplify contour
    epsilon = simplify_tolerance
    contour = cv2.approxPolyDP(contour, epsilon, True)

    # Convert to list of tuples
    if len(contour) < 3:  # Need at least 3 points for a polygon
        return None

    vertices = [(int(pt[0][0]), int(pt[0][1])) for pt in contour]

    return vertices


def polygon_to_yolo_format(vertices: List[Tuple[float, float]], img_w: int, img_h: int) -> List[float]:
    """
    Convert polygon vertices to YOLO segmentation format (normalized coordinates).

    Args:
        vertices: List of (x, y) pixel coordinates
        img_w: Image width
        img_h: Image height

    Returns:
        Flattened list of normalized coordinates: [x1, y1, x2, y2, ...]
    """
    normalized = []
    for x, y in vertices:
        x_norm = x / img_w
        y_norm = y / img_h
        # Clip to [0, 1]
        x_norm = max(0.0, min(1.0, x_norm))
        y_norm = max(0.0, min(1.0, y_norm))
        normalized.extend([x_norm, y_norm])

    return normalized


def write_yolo_segmentation(output_path: Path, class_id: int, normalized_coords: List[float]):
    """
    Write YOLO segmentation format: class_id x1 y1 x2 y2 x3 y3 ...
    """
    with open(output_path, 'a') as f:
        coords_str = ' '.join(f'{c:.6f}' for c in normalized_coords)
        f.write(f"{class_id} {coords_str}\n")


def process_image(
    image_path: Path,
    label_path: Path,
    output_path: Path,
    sam_model,
    sam_processor,
    conf_threshold: float,
    simplify_tolerance: float,
    device: str,
    is_sam2: bool = False
) -> Tuple[int, int]:
    """
    Process a single image: read detections, run SAM/SAM2, save segmentation polygons.

    Returns:
        (num_detections, num_segmentations)
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    img_w, img_h = image.size

    # Read detection labels
    if not label_path.exists():
        return 0, 0

    with open(label_path, 'r') as f:
        lines = f.readlines()

    if not lines:
        return 0, 0

    # Parse detections
    detections = []
    for line in lines:
        try:
            class_id, cx, cy, w, h, conf = parse_yolo_detection(line)
            if conf >= conf_threshold:
                detections.append((class_id, cx, cy, w, h, conf))
        except ValueError as e:
            warnings.warn(f"Skipping invalid line in {label_path}: {e}")
            continue

    if not detections:
        return len(lines), 0

    # Convert to bboxes for SAM
    input_boxes = []
    class_ids = []
    for class_id, cx, cy, w, h, conf in detections:
        bbox = yolo_to_xyxy(cx, cy, w, h, img_w, img_h)
        input_boxes.append(bbox)
        class_ids.append(class_id)

    # Prepare SAM inputs
    # Format: [image level, box level, box coordinates]
    inputs = sam_processor(
        image,
        input_boxes=[input_boxes],  # 3 levels: [boxes_for_image, individual_box, coordinates]
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Run SAM inference
    with torch.no_grad():
        outputs = sam_model(**inputs)

    # SAM outputs multiple mask candidates per box
    # pred_masks shape: [batch=1, num_boxes, num_candidates=3, H=256, W=256]
    # iou_scores shape: [batch=1, num_boxes, num_candidates=3]
    pred_masks = outputs.pred_masks[0]  # Remove batch dimension: [num_boxes, 3, 256, 256]
    iou_scores = outputs.iou_scores[0]   # Remove batch dimension: [num_boxes, 3]

    # Select best mask for each box (highest IoU score)
    best_mask_indices = iou_scores.argmax(dim=1)  # [num_boxes]

    # Clear output file
    if output_path.exists():
        output_path.unlink()

    # Convert masks to polygons
    num_segmentations = 0
    for i, (class_id, best_idx) in enumerate(zip(class_ids, best_mask_indices)):
        # Get the best mask for this box
        mask_logits = pred_masks[i, best_idx]  # Shape: [256, 256]

        # Resize mask to original image size using interpolation
        mask_logits_resized = torch.nn.functional.interpolate(
            mask_logits.unsqueeze(0).unsqueeze(0),  # [1, 1, 256, 256]
            size=(img_h, img_w),
            mode='bilinear',
            align_corners=False
        ).squeeze()  # [H, W]

        # Apply sigmoid and threshold to get binary mask
        mask_prob = torch.sigmoid(mask_logits_resized)
        mask_np = (mask_prob > 0.5).cpu().numpy()  # Binarize

        # Convert to polygon
        vertices = mask_to_polygon(mask_np, simplify_tolerance=simplify_tolerance)

        if vertices is None or len(vertices) < 3:
            warnings.warn(f"Skipping invalid polygon for detection {i} in {image_path.name}")
            continue

        # Convert to YOLO format
        normalized_coords = polygon_to_yolo_format(vertices, img_w, img_h)

        # Write to file
        write_yolo_segmentation(output_path, class_id, normalized_coords)
        num_segmentations += 1

    return len(detections), num_segmentations


def find_images_and_labels(
    images_dir: Path,
    labels_dir: Path
) -> List[Tuple[Path, Path]]:
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
        description='Auto-segmentation using RT-DETR detections + SAM2'
    )
    parser.add_argument('--images', type=str, required=True,
                        help='Directory containing images')
    parser.add_argument('--labels', type=str, required=True,
                        help='Directory containing YOLO detection labels (from infer_hf.py --save-txt)')
    parser.add_argument('--outdir', type=str, required=True,
                        help='Output directory for segmentation labels')
    parser.add_argument('--sam-model', type=str, default='facebook/sam2-hiera-tiny',
                        help='SAM model from HuggingFace (default: facebook/sam2-hiera-tiny). '
                             'SAM2: sam2-hiera-tiny (fastest), sam2-hiera-small, sam2-hiera-base-plus, sam2.1-hiera-large (best). '
                             'SAM1: sam-vit-base, sam-vit-large, sam-vit-huge')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold for detections (default: 0.25)')
    parser.add_argument('--simplify', type=float, default=1.5,
                        help='Polygon simplification tolerance in pixels (default: 1.5)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda or cpu (default: cuda)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size (currently only 1 supported)')

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

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        device = torch.device('cpu')

    print(f"\n🚀 Auto-segmentation with SAM")
    print(f"   Images:     {images_dir}")
    print(f"   Labels:     {labels_dir}")
    print(f"   Output:     {outdir}")
    print(f"   SAM Model:  {args.sam_model}")
    print(f"   Device:     {device}")
    print(f"   Confidence: {args.conf}")
    print(f"   Simplify:   {args.simplify} pixels")

    # Load SAM model and processor (auto-detect SAM1 vs SAM2)
    print(f"\n📦 Loading SAM model: {args.sam_model}...")
    try:
        # Detect if SAM2 based on model name
        is_sam2 = 'sam2' in args.sam_model.lower()

        if is_sam2:
            print("   Detected SAM2 model")
            sam_processor = Sam2Processor.from_pretrained(args.sam_model)
            sam_model = Sam2Model.from_pretrained(args.sam_model)
        else:
            print("   Detected SAM1 model")
            sam_processor = AutoProcessor.from_pretrained(args.sam_model)
            sam_model = SamModel.from_pretrained(args.sam_model)

        sam_model.to(device)
        sam_model.eval()
        print(f"✅ SAM model loaded successfully ({type(sam_model).__name__})")
    except Exception as e:
        print(f"❌ Failed to load SAM model: {e}")
        print("\nAvailable models:")
        print("  SAM2 (recommended):")
        print("    - facebook/sam2-hiera-tiny (fastest)")
        print("    - facebook/sam2-hiera-small")
        print("    - facebook/sam2-hiera-base-plus")
        print("    - facebook/sam2.1-hiera-large (most accurate)")
        print("  SAM1 (legacy):")
        print("    - facebook/sam-vit-base")
        print("    - facebook/sam-vit-large")
        print("    - facebook/sam-vit-huge")
        sys.exit(1)

    # Find image-label pairs
    print("\n🔍 Finding images with detection labels...")
    pairs = find_images_and_labels(images_dir, labels_dir)

    if not pairs:
        print(f"❌ No images with corresponding labels found!")
        print(f"   Make sure you ran: infer_hf.py --save-txt")
        sys.exit(1)

    print(f"✅ Found {len(pairs)} images with detection labels")

    # Process images
    print(f"\n🎯 Processing images...")
    total_detections = 0
    total_segmentations = 0

    for img_path, label_path in tqdm(pairs, desc="Auto-segmenting"):
        # Determine output path
        rel_path = label_path.relative_to(labels_dir)
        output_path = outdir / rel_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Process
        num_det, num_seg = process_image(
            img_path,
            label_path,
            output_path,
            sam_model,
            sam_processor,
            args.conf,
            args.simplify,
            device,
            is_sam2=is_sam2
        )

        total_detections += num_det
        total_segmentations += num_seg

    # Summary
    print(f"\n✅ Auto-segmentation complete!")
    print(f"   Total detections:    {total_detections}")
    print(f"   Total segmentations: {total_segmentations}")
    print(f"   Success rate:        {total_segmentations/total_detections*100:.1f}%")
    print(f"\n📁 Segmentation labels saved to: {outdir}")
    print(f"\n📊 Next step: Convert to geospatial polygons:")
    print(f"   python convert_yolo_seg_to_gpkg.py \\")
    print(f"     --labels {outdir} \\")
    print(f"     --images {images_dir} \\")
    print(f"     --output pools_segmented.gpkg")


if __name__ == '__main__':
    main()
