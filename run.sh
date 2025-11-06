#!/bin/bash
# RT-DETR Training Script
# Improved parameters based on dataset analysis:
# - Dataset: 1,284 train images, 385 annotations (~0.3 objects/image)
# - Reduced epochs to prevent overfitting
# - Reduced batch size to prevent OOM
# - Evaluates validation loss at each epoch (default behavior)
# - COCO mAP metrics computed after training (see evaluate_hf.py below)

python train_hf.py \
  --train-json datasets/coco_format/train_coco.json \
  --val-json datasets/coco_format/val_coco.json \
  --model PekingU/rtdetr_v2_r34vd \
  --output runs/train/rtdetr_v2/improved \
  --epochs 20 \
  --batch-size 4 \
  --lr 1e-4 \
  --warmup-steps 100

# Optional flags:
#  --eval-strategy no          # Disable all evaluation during training
#  --enable-coco-eval          # Enable COCO mAP metrics during training (EXPERIMENTAL)

# After training completes, evaluate with COCO metrics:
# python evaluate_hf.py \
#   --model runs/train/rtdetr_v2/improved/final_model \
#   --val-json datasets/coco_format/val_coco.json \
#   --conf 0.25
