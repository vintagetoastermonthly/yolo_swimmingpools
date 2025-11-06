#!/usr/bin/env python3
"""
Convert YOLO-format dataset to COCO JSON format for HuggingFace RT-DETR training.

Usage:
  python yolo_to_coco.py --data datasets/SwimmingPools_Auckland_090925/data.yaml --output datasets/coco_format
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import yaml
from PIL import Image
from tqdm import tqdm


def parse_yolo_label(label_path: Path) -> List[Dict]:
    """Parse YOLO format label file: class cx cy w h (normalized)"""
    annotations = []
    if not label_path.exists():
        return annotations

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(float(parts[0]))
                cx, cy, w, h = map(float, parts[1:5])
                annotations.append({
                    'class_id': class_id,
                    'bbox_norm': [cx, cy, w, h]
                })
    return annotations


def yolo_to_coco_bbox(cx: float, cy: float, w: float, h: float, img_w: int, img_h: int) -> List[float]:
    """
    Convert YOLO normalized bbox (cx, cy, w, h) to COCO format (x, y, width, height) in pixels.
    COCO uses top-left corner (x, y) + width, height.
    """
    # Denormalize
    cx_px = cx * img_w
    cy_px = cy * img_h
    w_px = w * img_w
    h_px = h * img_h

    # Convert center format to top-left corner format
    x = cx_px - w_px / 2
    y = cy_px - h_px / 2

    return [x, y, w_px, h_px]


def convert_split(image_list_path: Path, labels_root: Path, class_names: Dict[int, str],
                  output_json: Path, images_root: Path):
    """Convert one split (train or val) to COCO format"""

    # Read image paths
    with open(image_list_path, 'r') as f:
        image_paths = [line.strip() for line in f if line.strip()]

    coco_data = {
        'info': {
            'description': 'YOLO to COCO converted dataset',
            'version': '1.0',
            'year': 2024,
            'contributor': '',
            'date_created': ''
        },
        'images': [],
        'annotations': [],
        'categories': [{'id': int(k), 'name': v} for k, v in class_names.items()]
    }

    annotation_id = 1

    for img_id, img_path_str in enumerate(tqdm(image_paths, desc=f"Converting {image_list_path.name}"), start=1):
        img_path = Path(img_path_str)

        # Get image dimensions
        try:
            with Image.open(img_path) as img:
                img_w, img_h = img.size
        except Exception as e:
            print(f"Warning: Could not open {img_path}: {e}")
            continue

        # Add image entry
        coco_data['images'].append({
            'id': img_id,
            'file_name': str(img_path),
            'width': img_w,
            'height': img_h
        })

        # Find corresponding label file
        # Labels mirror the image directory structure, with "images" replaced by "labels"
        img_path_str = str(img_path)
        if '/images/' in img_path_str:
            label_path_str = img_path_str.replace('/images/', '/labels/').replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
            label_path = Path(label_path_str)
        else:
            # Fallback: try simple stem match
            label_path = labels_root / img_path.stem
            if not label_path.suffix:
                label_path = label_path.with_suffix('.txt')

            # Try to find label in various locations
            if not label_path.exists():
                # Try relative to labels_root
                label_path = labels_root / (img_path.stem + '.txt')

        # Parse annotations
        yolo_annotations = parse_yolo_label(label_path)

        for ann in yolo_annotations:
            cx, cy, w, h = ann['bbox_norm']
            bbox_coco = yolo_to_coco_bbox(cx, cy, w, h, img_w, img_h)

            coco_data['annotations'].append({
                'id': annotation_id,
                'image_id': img_id,
                'category_id': ann['class_id'],
                'bbox': bbox_coco,
                'area': bbox_coco[2] * bbox_coco[3],
                'iscrowd': 0
            })
            annotation_id += 1

    # Write COCO JSON
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(coco_data, f, indent=2)

    print(f"✅ Wrote {len(coco_data['images'])} images and {len(coco_data['annotations'])} annotations to {output_json}")


def main():
    parser = argparse.ArgumentParser(description="Convert YOLO dataset to COCO format")
    parser.add_argument('--data', type=str, required=True, help='Path to YOLO data.yaml')
    parser.add_argument('--output', type=str, required=True, help='Output directory for COCO JSON files')
    parser.add_argument('--labels-root', type=str, default=None,
                       help='Root directory for label files (auto-detected if not provided)')
    args = parser.parse_args()

    # Load YOLO data.yaml
    data_yaml = Path(args.data)
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)

    # Extract info
    class_names = data_config['names']
    train_txt = Path(data_config['train'])
    val_txt = Path(data_config['val'])

    # Determine labels root
    if args.labels_root:
        labels_root = Path(args.labels_root)
    else:
        # Auto-detect: typically labels/train/ or labels/val/
        labels_root = data_yaml.parent / 'labels'

    output_dir = Path(args.output)

    # Convert train split
    if train_txt.exists():
        train_labels = labels_root / 'train'
        if not train_labels.exists():
            train_labels = labels_root
        convert_split(
            train_txt,
            train_labels,
            class_names,
            output_dir / 'train_coco.json',
            data_yaml.parent
        )

    # Convert val split
    if val_txt.exists():
        val_labels = labels_root / 'val'
        if not val_labels.exists():
            val_labels = labels_root / 'train'  # Sometimes val labels are in train folder
        if not val_labels.exists():
            val_labels = labels_root
        convert_split(
            val_txt,
            val_labels,
            class_names,
            output_dir / 'val_coco.json',
            data_yaml.parent
        )

    print("\n✅ Conversion complete!")
    print(f"Train annotations: {output_dir / 'train_coco.json'}")
    print(f"Val annotations: {output_dir / 'val_coco.json'}")


if __name__ == '__main__':
    main()
