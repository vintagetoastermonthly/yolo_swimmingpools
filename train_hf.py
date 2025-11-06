#!/usr/bin/env python3
"""
RT-DETR/RT-DETRv2 training using HuggingFace Transformers (Apache 2.0 licensed).
Completely clean implementation with no proprietary licensing concerns.

Usage:
  python train_hf.py --train-json datasets/coco_format/train_coco.json \
                     --val-json datasets/coco_format/val_coco.json \
                     --model PekingU/rtdetr_v2_r34vd \
                     --output runs/train/rtdetr_v2 \
                     --epochs 100 \
                     --batch-size 16
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from transformers import (
    AutoModelForObjectDetection,
    AutoImageProcessor,
    TrainingArguments,
    Trainer
)
from transformers.image_transforms import center_to_corners_format
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class COCODetectionDataset(Dataset):
    """PyTorch Dataset for COCO format object detection"""

    def __init__(self, json_path: str, processor, transforms=None):
        self.processor = processor
        self.transforms = transforms

        # Load COCO JSON
        with open(json_path, 'r') as f:
            coco_data = json.load(f)

        self.images = coco_data['images']
        self.annotations = coco_data['annotations']

        # Create category ID mapping
        self.categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
        self.num_classes = len(self.categories)

        # Group annotations by image_id
        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info['id']
        img_path = img_info['file_name']

        # Load image
        image = Image.open(img_path).convert('RGB')
        img_w, img_h = image.size

        # Get annotations for this image
        anns = self.img_to_anns.get(img_id, [])

        # Prepare target in COCO format
        boxes = []
        labels = []
        areas = []
        for ann in anns:
            # COCO bbox format: [x, y, width, height]
            x, y, w, h = ann['bbox']
            # Convert to [x_min, y_min, x_max, y_max] for albumentations
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
            areas.append(w * h)

        # Apply augmentations if provided
        if self.transforms:
            # Convert to albumentations format
            transformed = self.transforms(
                image=np.array(image),
                bboxes=boxes,
                category_ids=labels
            )
            image = Image.fromarray(transformed['image'])
            boxes = transformed['bboxes']
            labels = transformed['category_ids']
            # Recalculate areas after transformation
            areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]

        # Convert boxes back to COCO format [x, y, w, h]
        coco_boxes = []
        for box in boxes:
            x_min, y_min, x_max, y_max = box
            coco_boxes.append([x_min, y_min, x_max - x_min, y_max - y_min])

        # Prepare target dict in proper COCO format
        annotations = []
        for bbox, cat_id, area in zip(coco_boxes, labels, areas):
            annotations.append({
                'bbox': bbox,
                'category_id': int(cat_id),
                'area': area,
                'iscrowd': 0
            })

        target = {
            'image_id': img_id,
            'annotations': annotations
        }

        # Process with HuggingFace processor
        encoding = self.processor(images=image, annotations=target, return_tensors="pt")

        # Remove batch dimension (added by processor)
        pixel_values = encoding['pixel_values'].squeeze(0)
        labels = encoding['labels'][0]

        return {
            'pixel_values': pixel_values,
            'labels': labels
        }


def collate_fn(batch):
    """Custom collate function for batching"""
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    labels = [item['labels'] for item in batch]

    return {
        'pixel_values': pixel_values,
        'labels': labels
    }


def get_train_transforms():
    """Training data augmentation pipeline"""
    # Disabled for now due to floating point precision issues with bbox coordinates
    # Can be re-enabled with proper bbox clipping
    return None
    # return A.Compose([
    #     A.HorizontalFlip(p=0.5),
    #     A.RandomBrightnessContrast(p=0.3),
    # ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids'], min_area=1, min_visibility=0.1))


def get_val_transforms():
    """Validation transforms (minimal)"""
    return None  # No augmentation for validation


class CocoEvaluator:
    """COCO evaluator for computing mAP metrics during training"""

    def __init__(self, coco_gt, iou_types=['bbox']):
        """
        Args:
            coco_gt: COCO ground truth object
            iou_types: List of IoU types to evaluate (default: ['bbox'])
        """
        self.coco_gt = coco_gt
        self.iou_types = iou_types
        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)
        self.img_ids = []
        self.eval_imgs = []
        self.predictions = []

    def update(self, predictions):
        """
        Args:
            predictions: List of dicts with keys: image_id, category_id, bbox, score
        """
        self.predictions.extend(predictions)

    def synchronize_between_processes(self):
        """Placeholder for distributed training sync"""
        pass

    def accumulate(self):
        """Accumulate predictions and compute metrics"""
        for iou_type in self.iou_types:
            coco_eval = self.coco_eval[iou_type]
            if len(self.predictions) > 0:
                # Load predictions into COCO format
                coco_dt = self.coco_gt.loadRes(self.predictions)
                coco_eval.cocoDt = coco_dt
                coco_eval.evaluate()
                coco_eval.accumulate()

    def summarize(self):
        """Compute and print summary statistics"""
        results = {}
        for iou_type in self.iou_types:
            coco_eval = self.coco_eval[iou_type]
            coco_eval.summarize()
            # Extract key metrics
            results[f'{iou_type}_mAP'] = coco_eval.stats[0]  # mAP @ IoU=0.50:0.95
            results[f'{iou_type}_mAP50'] = coco_eval.stats[1]  # mAP @ IoU=0.50
            results[f'{iou_type}_mAP75'] = coco_eval.stats[2]  # mAP @ IoU=0.75
        return results


def compute_metrics_factory(processor, val_coco_json):
    """Factory function to create compute_metrics with access to COCO GT"""

    # Load COCO ground truth
    coco_gt = COCO(val_coco_json)

    def compute_metrics(eval_pred):
        """
        Compute COCO detection metrics (mAP, mAP50, mAP75)

        Args:
            eval_pred: EvalPrediction with predictions and label_ids

        Returns:
            Dictionary with mAP metrics
        """
        # Get predictions from model
        logits, labels = eval_pred.predictions, eval_pred.label_ids

        # Convert model outputs to COCO format
        predictions = []

        # Process each prediction
        for idx, (pred_logits, pred_boxes) in enumerate(zip(logits[0], logits[1])):
            # pred_logits shape: [num_queries, num_classes]
            # pred_boxes shape: [num_queries, 4]

            # Get scores and labels
            scores = pred_logits.softmax(-1)[:, :-1]  # Exclude background class
            scores, pred_labels = scores.max(-1)

            # Filter by confidence threshold
            keep = scores > 0.05
            scores = scores[keep]
            pred_labels = pred_labels[keep]
            pred_boxes = pred_boxes[keep]

            # Convert boxes from normalized [cx, cy, w, h] to COCO format [x, y, w, h]
            # Get image size from labels
            if idx < len(labels):
                image_id = labels[idx]['image_id'].item() if torch.is_tensor(labels[idx]['image_id']) else labels[idx]['image_id']
                img_info = coco_gt.loadImgs(image_id)[0]
                img_h, img_w = img_info['height'], img_info['width']

                # Denormalize boxes
                boxes = center_to_corners_format(pred_boxes)
                boxes = boxes * torch.tensor([img_w, img_h, img_w, img_h], device=boxes.device)

                # Convert to [x, y, w, h]
                boxes[:, 2:] = boxes[:, 2:] - boxes[:, :2]

                # Add predictions
                for box, score, label in zip(boxes.cpu().numpy(), scores.cpu().numpy(), pred_labels.cpu().numpy()):
                    predictions.append({
                        'image_id': int(image_id),
                        'category_id': int(label),
                        'bbox': box.tolist(),
                        'score': float(score)
                    })

        # Evaluate using COCO API
        if len(predictions) == 0:
            return {
                'mAP': 0.0,
                'mAP50': 0.0,
                'mAP75': 0.0
            }

        evaluator = CocoEvaluator(coco_gt, iou_types=['bbox'])
        evaluator.update(predictions)
        evaluator.accumulate()
        results = evaluator.summarize()

        return {
            'mAP': results['bbox_mAP'],
            'mAP50': results['bbox_mAP50'],
            'mAP75': results['bbox_mAP75']
        }

    return compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Train RT-DETR using HuggingFace Transformers')
    parser.add_argument('--train-json', type=str, required=True, help='Path to train COCO JSON')
    parser.add_argument('--val-json', type=str, required=True, help='Path to validation COCO JSON')
    parser.add_argument('--model', type=str, default='PekingU/rtdetr_v2_r34vd',
                       help='HuggingFace model name (PekingU/rtdetr_v2_r34vd, PekingU/rtdetr_v2_r50vd, etc.)')
    parser.add_argument('--output', type=str, default='runs/train/rtdetr_hf', help='Output directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Per-device batch size')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--warmup-steps', type=int, default=300, help='Warmup steps')
    parser.add_argument('--eval-strategy', type=str, default='epoch', help='Evaluation strategy (no/epoch/steps). Default: epoch (evaluates validation loss). Add --enable-coco-eval for mAP metrics.')
    parser.add_argument('--save-steps', type=int, default=500, help='Save checkpoint every N steps')
    parser.add_argument('--device', type=str, default='cuda', help='Device: cuda or cpu')
    parser.add_argument('--enable-coco-eval', action='store_true', help='Enable COCO mAP evaluation during training (EXPERIMENTAL - may cause errors). Without this flag, only validation loss is computed.')
    return parser.parse_args()


def main():
    args = parse_args()

    # Get number of classes from train data first
    with open(args.train_json, 'r') as f:
        train_coco = json.load(f)
    num_classes = len(train_coco['categories'])

    # Create id2label and label2id mappings
    id2label = {cat['id']: cat['name'] for cat in train_coco['categories']}
    label2id = {v: k for k, v in id2label.items()}

    # Load processor and model
    print(f"Loading model: {args.model}")
    processor = AutoImageProcessor.from_pretrained(args.model, use_fast=True)

    # Load model with proper class configuration
    model = AutoModelForObjectDetection.from_pretrained(
        args.model,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True  # Allow resizing of classification head
    )

    print(f"Model loaded with {num_classes} classes")

    # Create datasets
    print("Creating datasets...")
    train_dataset = COCODetectionDataset(
        args.train_json,
        processor,
        transforms=get_train_transforms()
    )

    val_dataset = COCODetectionDataset(
        args.val_json,
        processor,
        transforms=get_val_transforms()
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create compute_metrics function with access to validation COCO GT
    compute_metrics = None
    if args.enable_coco_eval:
        print("⚠️  WARNING: COCO mAP evaluation during training is EXPERIMENTAL and may cause errors!")
        print("Setting up COCO evaluation metrics (mAP, mAP50, mAP75)...")
        compute_metrics = compute_metrics_factory(processor, args.val_json)
    else:
        if args.eval_strategy != 'no':
            print(f"ℹ️  Evaluation strategy: {args.eval_strategy} (validation loss only)")
            print("   For COCO mAP metrics, use evaluate_hf.py after training or add --enable-coco-eval")
        else:
            print("ℹ️  Evaluation disabled (--eval-strategy no)")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        eval_strategy=args.eval_strategy,
        save_strategy='epoch',
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=(args.enable_coco_eval and args.eval_strategy != 'no'),
        metric_for_best_model='mAP50' if args.enable_coco_eval else None,
        greater_is_better=True if args.enable_coco_eval else None,
        logging_dir=f"{args.output}/logs",
        logging_steps=50,
        remove_unused_columns=False,
        push_to_hub=False,
        dataloader_num_workers=4,
        fp16=torch.cuda.is_available(),  # Mixed precision if GPU available
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        tokenizer=processor,  # Pass processor as tokenizer for saving
        compute_metrics=compute_metrics  # None if evaluation disabled
    )

    # Train
    print("\n🚀 Starting training...")
    trainer.train()

    # Save final model
    print("\n💾 Saving final model...")
    trainer.save_model(f"{args.output}/final_model")
    processor.save_pretrained(f"{args.output}/final_model")

    print(f"\n✅ Training complete! Model saved to {args.output}/final_model")

    # Suggest evaluation if it wasn't done during training
    if not args.enable_coco_eval:
        print("\n📊 To evaluate your model with COCO metrics, run:")
        print(f"   python evaluate_hf.py --model {args.output}/final_model --val-json {args.val_json}")


if __name__ == '__main__':
    # Import numpy here to avoid issues if not used
    import numpy as np
    main()
