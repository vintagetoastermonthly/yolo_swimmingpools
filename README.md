# RT-DETR Swimming Pools Detection

Object detection for swimming pools in Auckland aerial imagery using **RT-DETR** via **HuggingFace Transformers** (Apache 2.0 licensed - commercial-friendly).

## Why RT-DETR + HuggingFace?
- **Apache 2.0 License**: Fully permissive for commercial use and selling derived data
- **Zero proprietary dependencies**: No Ultralytics, no YOLO, no AGPL-3.0 concerns
- **Clean licensing stack**: HuggingFace Transformers (Apache 2.0) + RT-DETR (Apache 2.0)
- **No legal questions**: Completely avoids any association with restrictive licenses
- **State-of-the-art performance**: Transformer-based architecture, 53-55% AP on COCO
- **Production-ready**: HuggingFace is used by thousands of companies for commercial AI

## Quick Start

```bash
# 1. Convert YOLO dataset to COCO format
python yolo_to_coco.py --data datasets/SwimmingPools_Auckland_090925/data.yaml --output datasets/coco_format

# 2. Train RT-DETR
python train_hf.py --train-json datasets/coco_format/train_coco.json \
                   --val-json datasets/coco_format/val_coco.json \
                   --model PekingU/rtdetr_r50vd \
                   --output runs/train/rtdetr_hf

# 3. Run inference
python infer_hf.py --model runs/train/rtdetr_hf/final_model \
                   --source "images/*.jpg" \
                   --save-txt --save-img
```

See `CLAUDE.md` for detailed documentation and commands.