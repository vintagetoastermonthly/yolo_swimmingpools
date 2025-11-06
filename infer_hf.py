#!/usr/bin/env python3
"""
RT-DETR inference using HuggingFace Transformers (Apache 2.0 licensed).
Memory-safe streaming inference with optional sampling and multiple output formats.

Usage:
  python infer_hf.py --model runs/train/rtdetr_hf/final_model \
                     --source "images/*.jpg" \
                     --outdir runs/infer \
                     --conf 0.25 \
                     --save-txt --save-json detections.ndjson
"""

import argparse
import glob
import json
import random
import sys
from pathlib import Path
from typing import Iterator, List, Optional
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForObjectDetection, AutoImageProcessor
from tqdm.auto import tqdm


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def iter_image_paths_from_listfile(listfile: Path) -> Iterator[str]:
    """Iterate over image paths from a text file"""
    with open(listfile, 'r') as f:
        for line in f:
            path = line.strip()
            if path and Path(path).suffix.lower() in IMG_EXTS:
                yield path


def iter_image_paths_from_dir(dirpath: Path) -> Iterator[str]:
    """Recursively iterate over images in a directory"""
    for ext in IMG_EXTS:
        for f in dirpath.rglob(f"*{ext}"):
            if f.is_file():
                yield str(f)


def iter_image_paths_from_glob(glob_pattern: str) -> Iterator[str]:
    """Iterate over images matching a glob pattern"""
    for path in glob.iglob(glob_pattern, recursive=True):
        if Path(path).suffix.lower() in IMG_EXTS and Path(path).is_file():
            yield str(path)


def reservoir_sample(iterable, k: int, seed: int = 42) -> List[str]:
    """Reservoir sampling: uniformly sample k items from an iterable"""
    rng = random.Random(seed)
    sample = []
    for i, item in enumerate(iterable, start=1):
        if i <= k:
            sample.append(item)
        else:
            j = rng.randint(1, i)
            if j <= k:
                sample[j - 1] = item
    return sample


def get_image_paths(source: str, sample_size: int = 0, seed: int = 42) -> List[str]:
    """
    Get list of image paths from various source types.
    Supports: single file, directory, glob pattern, or .txt list file.
    """
    source_path = Path(source)

    # Single file
    if source_path.is_file():
        if source_path.suffix.lower() == '.txt':
            # List file
            iterator = iter_image_paths_from_listfile(source_path)
        elif source_path.suffix.lower() in IMG_EXTS:
            return [str(source_path)]
        else:
            raise ValueError(f"Unsupported file type: {source_path}")
    # Directory
    elif source_path.is_dir():
        iterator = iter_image_paths_from_dir(source_path)
    # Glob pattern
    elif any(ch in source for ch in '*?[]'):
        iterator = iter_image_paths_from_glob(source)
    else:
        raise ValueError(f"Invalid source: {source}")

    # Apply sampling if requested
    if sample_size > 0:
        return reservoir_sample(iterator, sample_size, seed)
    else:
        return list(iterator)


def xyxy_to_xywh_norm(bbox: List[float], img_w: int, img_h: int) -> List[float]:
    """Convert [x1, y1, x2, y2] to normalized [cx, cy, w, h]"""
    x1, y1, x2, y2 = bbox
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return [cx, cy, w, h]


def draw_detections(image: Image.Image, boxes, labels, scores, class_names, threshold=0.25):
    """Draw bounding boxes on image"""
    draw = ImageDraw.Draw(image)

    # Try to load a font, fallback to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        font = ImageFont.load_default()

    for box, label, score in zip(boxes, labels, scores):
        if score < threshold:
            continue

        x1, y1, x2, y2 = box
        class_name = class_names.get(int(label), f"class_{label}")
        text = f"{class_name}: {score:.2f}"

        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

        # Draw label background
        text_bbox = draw.textbbox((x1, y1), text, font=font)
        draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2], fill="red")
        draw.text((x1, y1), text, fill="white", font=font)

    return image


def parse_args():
    parser = argparse.ArgumentParser(description='RT-DETR inference with HuggingFace')
    parser.add_argument('--model', type=str, required=True, help='Path to model directory or HF model name')
    parser.add_argument('--source', type=str, required=True,
                       help='Image source: file/dir/glob/"list.txt" (quote globs!)')
    parser.add_argument('--outdir', type=str, default='runs/infer', help='Output directory')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--device', type=str, default='cuda', help='Device: cuda or cpu')
    parser.add_argument('--save-img', action='store_true', help='Save annotated images')
    parser.add_argument('--save-txt', action='store_true', help='Save YOLO-format txt labels')
    parser.add_argument('--save-json', type=str, default='', help='Save NDJSON detections to file')
    parser.add_argument('--sample', type=int, default=0, help='Random sample size (0 = all)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for inference')
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")

    # Load model and processor
    print(f"Loading model from {args.model}...")
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForObjectDetection.from_pretrained(args.model)
    model.to(device)
    model.eval()

    # Get class names
    class_names = {i: f"class_{i}" for i in range(model.config.num_labels)}
    # Try to load from config
    if hasattr(model.config, 'id2label'):
        class_names = model.config.id2label

    # Setup output directories
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.save_img:
        img_outdir = outdir / 'images'
        img_outdir.mkdir(exist_ok=True)

    if args.save_txt:
        txt_outdir = outdir / 'labels'
        txt_outdir.mkdir(exist_ok=True)

    # Get image paths
    print(f"Gathering images from {args.source}...")
    image_paths = get_image_paths(args.source, args.sample, args.seed)

    if not image_paths:
        print("❌ No images found!")
        return

    print(f"Processing {len(image_paths)} images...")

    # Open JSON file if needed
    json_fp = open(args.save_json, 'w') if args.save_json else None

    # Process images
    for img_path_str in tqdm(image_paths, desc="Inference", unit="img"):
        img_path = Path(img_path_str)

        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Could not open {img_path}: {e}")
            continue

        img_w, img_h = image.size

        # Process with model
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process
        target_sizes = torch.tensor([(img_h, img_w)]).to(device)
        results = processor.post_process_object_detection(
            outputs,
            threshold=args.conf,
            target_sizes=target_sizes
        )[0]

        boxes = results['boxes'].cpu().numpy()
        scores = results['scores'].cpu().numpy()
        labels = results['labels'].cpu().numpy()

        # Save YOLO txt
        if args.save_txt:
            txt_path = txt_outdir / f"{img_path.stem}.txt"
            with open(txt_path, 'w') as f:
                for box, label, score in zip(boxes, labels, scores):
                    cx, cy, w, h = xyxy_to_xywh_norm(box.tolist(), img_w, img_h)
                    f.write(f"{int(label)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {score:.6f}\n")

        # Save NDJSON
        if json_fp:
            for box, label, score in zip(boxes, labels, scores):
                record = {
                    'image': str(img_path),
                    'bbox': box.tolist(),
                    'conf': float(score),
                    'class_id': int(label),
                    'class_name': class_names.get(int(label), f"class_{label}"),
                    'width': img_w,
                    'height': img_h
                }
                json_fp.write(json.dumps(record) + '\n')

        # Save annotated image
        if args.save_img:
            annotated = draw_detections(
                image.copy(),
                boxes,
                labels,
                scores,
                class_names,
                args.conf
            )
            annotated.save(img_outdir / f"{img_path.stem}_pred{img_path.suffix}")

        # Clear GPU cache periodically
        if device.type == 'cuda' and len(image_paths) > 100:
            torch.cuda.empty_cache()

    if json_fp:
        json_fp.close()

    print(f"\n✅ Done! Outputs in {outdir}")
    if args.save_img:
        print(f"  • Annotated images: {img_outdir}")
    if args.save_txt:
        print(f"  • YOLO labels: {txt_outdir}")
    if args.save_json:
        print(f"  • NDJSON: {Path(args.save_json).resolve()}")


if __name__ == '__main__':
    main()
