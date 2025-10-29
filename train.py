#!/usr/bin/env python3
"""
Tiny YOLOv11 training script for YOLO-format detection data.
- Uses Ultralytics API
- Works on CPU or GPU
- Trains, validates, and exports a ready-to-use model

Usage:
  python train_yolo11.py --data dataset.yaml --model yolo11n.pt --epochs 100 --imgsz 640
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv11 on YOLO-format data")
    p.add_argument("--data", type=str, required=True, help="Path to dataset.yaml")
    p.add_argument("--model", type=str, default="yolo11n.pt",
                   help="YOLOv11 model checkpoint (e.g., yolo11n.pt, yolo11s.pt, yolo11m.pt...)")
    p.add_argument("--epochs", type=int, default=100, help="Training epochs")
    p.add_argument("--imgsz", type=int, default=640, help="Image size")
    p.add_argument("--batch", type=int, default=16, help="Batch size (auto if 0)")
    p.add_argument("--device", type=str, default="0", help="Device: 'auto', 'cpu', '0', '0,1'")
    p.add_argument("--project", type=str, default="runs/train", help="Project dir for runs")
    p.add_argument("--name", type=str, default="yolo11", help="Run name")
    p.add_argument("--workers", type=int, default=8, help="Dataloader workers")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--resume", action="store_true", help="Resume last training for this run name")
    p.add_argument("--patience", type=int, default=50, help="Early-stopping patience (epochs with no improvement)")
    return p.parse_args()

def main():
    args = parse_args()

    # Ensure paths exist
    data_yaml = Path(args.data).resolve()
    assert data_yaml.exists(), f"dataset yaml not found: {data_yaml}"

    # Load model (downloaded automatically on first use if not present)
    model = YOLO(args.model)

    # Train
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch if args.batch > 0 else 0,  # 0 = auto
        device=args.device,
        project=args.project,
        name=args.name,
        workers=args.workers,
        seed=args.seed,
        resume=args.resume,
        amp=True,              # mixed precision when available
        patience=args.patience # early stop
        # You can pass more knobs here (lr0, lrf, mosaic, hsv_h, flipud, etc.)
    )

    # Validate (mAP, PR curves, confusion matrix, etc.)
    metrics = model.val(
        data=str(data_yaml),
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=f"{args.name}-val"
    )
    print("Validation metrics:", metrics)

    # Export (pick one or many formats you actually need)
    # ONNX is broadly useful; add torchscript, openvino, coreml, etc. if desired.
    onnx_path = model.export(format="onnx", opset=12)
    print("Exported ONNX to:", onnx_path)

if __name__ == "__main__":
    main()

