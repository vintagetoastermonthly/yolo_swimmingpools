#!/usr/bin/env python3
"""
Evaluate RT-DETR/RT-DETRv2 model on validation set and compute COCO metrics.

Usage:
  python evaluate_hf.py --model runs/train/rtdetr_hf/final_model \
                        --val-json datasets/coco_format/val_coco.json \
                        --conf 0.25
"""

import argparse
import json
import torch
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from transformers import AutoModelForObjectDetection, AutoImageProcessor
from transformers.image_transforms import center_to_corners_format
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate RT-DETR model with COCO metrics')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model directory')
    parser.add_argument('--val-json', type=str, required=True, help='Path to validation COCO JSON')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--device', type=str, default='cuda', help='Device: cuda or cpu')
    return parser.parse_args()


def main():
    args = parse_args()

    # Load model and processor
    print(f"Loading model from: {args.model}")
    model = AutoModelForObjectDetection.from_pretrained(args.model)
    processor = AutoImageProcessor.from_pretrained(args.model)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Load COCO ground truth
    print(f"Loading validation data from: {args.val_json}")
    coco_gt = COCO(args.val_json)

    # Get all images
    img_ids = coco_gt.getImgIds()
    images = coco_gt.loadImgs(img_ids)

    print(f"Evaluating on {len(images)} images...")

    # Collect predictions
    predictions = []

    with torch.no_grad():
        for img_info in tqdm(images, desc="Running inference"):
            img_path = img_info['file_name']
            image = Image.open(img_path).convert('RGB')
            img_w, img_h = img_info['width'], img_info['height']
            image_id = img_info['id']

            # Preprocess image
            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Run inference
            outputs = model(**inputs)

            # Post-process outputs
            target_sizes = torch.tensor([[img_h, img_w]], device=device)
            results = processor.post_process_object_detection(
                outputs,
                threshold=args.conf,
                target_sizes=target_sizes
            )[0]

            # Convert to COCO format
            boxes = results['boxes'].cpu().numpy()
            scores = results['scores'].cpu().numpy()
            labels = results['labels'].cpu().numpy()

            for box, score, label in zip(boxes, scores, labels):
                # Convert from [x_min, y_min, x_max, y_max] to [x, y, w, h]
                x_min, y_min, x_max, y_max = box
                w = x_max - x_min
                h = y_max - y_min

                predictions.append({
                    'image_id': int(image_id),
                    'category_id': int(label),
                    'bbox': [float(x_min), float(y_min), float(w), float(h)],
                    'score': float(score)
                })

    print(f"\nTotal predictions: {len(predictions)}")

    if len(predictions) == 0:
        print("❌ No predictions made! Model may not be working correctly.")
        return

    # Run COCO evaluation
    print("\n" + "="*60)
    print("COCO Evaluation Results")
    print("="*60)

    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Extract key metrics
    print("\n" + "="*60)
    print("Summary Metrics")
    print("="*60)
    print(f"mAP (IoU=0.50:0.95): {coco_eval.stats[0]:.4f}")
    print(f"mAP50 (IoU=0.50):    {coco_eval.stats[1]:.4f}")
    print(f"mAP75 (IoU=0.75):    {coco_eval.stats[2]:.4f}")
    print(f"mAP (small):         {coco_eval.stats[3]:.4f}")
    print(f"mAP (medium):        {coco_eval.stats[4]:.4f}")
    print(f"mAP (large):         {coco_eval.stats[5]:.4f}")
    print("="*60)

    # Save results
    results_file = Path(args.model).parent / 'evaluation_results.json'
    results_dict = {
        'model': args.model,
        'val_json': args.val_json,
        'confidence_threshold': args.conf,
        'num_predictions': len(predictions),
        'num_images': len(images),
        'metrics': {
            'mAP': float(coco_eval.stats[0]),
            'mAP50': float(coco_eval.stats[1]),
            'mAP75': float(coco_eval.stats[2]),
            'mAP_small': float(coco_eval.stats[3]),
            'mAP_medium': float(coco_eval.stats[4]),
            'mAP_large': float(coco_eval.stats[5]),
        }
    }

    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"\n✅ Results saved to: {results_file}")


if __name__ == '__main__':
    main()
