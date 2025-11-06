#!/usr/bin/env python3
"""
RT-DETR inference (memory-safe, progress bar, optional random sampling)
Apache 2.0 licensed - commercial-friendly!

Features
- Accepts: file/dir/glob/"list.txt"/webcam/rtsp/http
- Memory-safe: streams results; optional reservoir sampling for large sources
- tqdm progress bar with correct totals
- Optional: save annotated media (--save-img), YOLO txt per image (--save-txt), NDJSON lines (--save-json)
- Speed knobs: --half, --workers, --device

Example:
  python infer_rtdetr.py --weights runs/train/rtdetr/weights/best.pt \
    --source "../chips_0075m/*.jpg" --sample 10000 --outdir runs/infer --conf 0.25 --half --workers 8
"""

import argparse
import glob
import json
import os
import random
import sys
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Union

import torch
from ultralytics import RTDETR
from tqdm.auto import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def parse_args():
    p = argparse.ArgumentParser("RT-DETR inference (memory-safe)")
    p.add_argument("--weights", type=str, required=True, help="Path to .pt (e.g., best.pt, rtdetr-l.pt)")
    p.add_argument("--source", type=str, required=True,
                   help="file/dir/glob/list(.txt)/webcam_index/rtsp/http (quote globs!)")
    p.add_argument("--outdir", type=str, default="runs/infer", help="Output directory")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    p.add_argument("--device", type=str, default="0", help="'auto', 'cpu', '0', '0,1', ...")
    p.add_argument("--maxdet", type=int, default=300, help="Max detections per image")
    p.add_argument("--save-img", action="store_true", help="Save annotated images/videos (opt-in)")
    p.add_argument("--save-txt", action="store_true", help="Save YOLO-format txt per-image (with conf)")
    p.add_argument("--save-json", type=str, default="", help="Write NDJSON detections to this file")
    p.add_argument("--half", action="store_true", help="Use FP16 on supported GPU for speed")
    p.add_argument("--vid-stride", type=int, default=1, help="Video frame stride")
    p.add_argument("--workers", type=int, default=4, help="Dataloader workers (where applicable)")
    p.add_argument("--show", action="store_true", help="Visualize in a window")
    p.add_argument("--sample", type=int, default=0, help="Random sample size (0 = use all)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    return p.parse_args()

def is_list_file(path: str) -> bool:
    return str(path).lower().endswith(".txt") and Path(path).is_file()

def iter_image_paths_from_listfile(listfile: Union[str, Path]) -> Iterator[str]:
    with open(listfile, "r") as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            p = Path(s)
            if p.suffix.lower() in IMG_EXTS and p.is_file():
                yield str(p)

def iter_image_paths_from_dir(dirpath: Union[str, Path]) -> Iterator[str]:
    p = Path(dirpath)
    for ext in IMG_EXTS:
        # rglob generator (no big lists)
        for f in p.rglob(f"*{ext}"):
            if f.is_file():
                yield str(f)

def iter_image_paths_from_glob(glob_str: str) -> Iterator[str]:
    # IMPORTANT: user should quote the glob in shell to prevent expansion
    for path in glob.iglob(glob_str, recursive=True):
        p = Path(path)
        if p.suffix.lower() in IMG_EXTS and p.is_file():
            yield str(p)

def is_stream_like(source: str) -> bool:
    return source.isdigit() or source.startswith(("rtsp://", "rtmp://", "http://", "https://"))

def count_iter(iterable: Iterable) -> int:
    """Count items in an iterable without storing them."""
    n = 0
    for _ in iterable:
        n += 1
    return n

def reservoir_sample(iterable: Iterable[str], k: int, seed: int = 42) -> List[str]:
    """
    Reservoir sampling: choose k items uniformly at random from an iterable of unknown/large size.
    O(N) time, O(k) memory.
    """
    rng = random.Random(seed)
    sample: List[str] = []
    for i, item in enumerate(iterable, start=1):
        if i <= k:
            sample.append(item)
        else:
            j = rng.randint(1, i)
            if j <= k:
                sample[j - 1] = item
    return sample

def build_sources_for_sampling(source: str) -> Optional[Iterator[str]]:
    """
    Return an iterator over candidate image paths for sampling, or None if not applicable.
    """
    p = Path(source)
    if is_list_file(source):
        return iter_image_paths_from_listfile(source)
    if p.is_dir():
        return iter_image_paths_from_dir(p)
    if any(ch in source for ch in "*?[]"):
        return iter_image_paths_from_glob(source)
    # single file, webcam, or stream: sampling not applicable here
    return None

def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load RT-DETR model
    model = RTDETR(args.weights)

    # Optional half precision for speed
    try:
        if args.half and torch.cuda.is_available() and hasattr(model, "model") and model.model is not None:
            model.model.half()
    except Exception:
        # Some variants may not expose .model.half(); safe to ignore
        pass

    names = getattr(model.model, "names", {})

    # Output dirs
    labels_dir = outdir / "labels"
    if args.save_txt:
        labels_dir.mkdir(parents=True, exist_ok=True)

    json_fp = open(args.save_json, "w", encoding="utf-8") if args.save_json else None

    # Determine source, possibly sampling
    src_for_ultralytics: Union[str, Sequence[str]] = args.source
    total: Optional[int] = None

    if args.sample > 0:
        iter_candidates = build_sources_for_sampling(args.source)
        if iter_candidates is not None:
            # Reservoir sample to avoid loading all paths into memory
            sampled = reservoir_sample(iter_candidates, args.sample, seed=args.seed)
            if len(sampled) == 0:
                print("No images found to sample.", file=sys.stderr)
                if json_fp:
                    json_fp.close()
                return
            src_for_ultralytics = sampled  # pass a list to Ultralytics
            total = len(sampled)
        else:
            # Not a sample-able source (e.g., webcam/single file). Use as-is.
            src_for_ultralytics = args.source

    if args.sample <= 0:
        # No sampling requested → we can provide an accurate total for progress in many cases
        if is_list_file(args.source):
            # Count without storing
            total = count_iter(iter_image_paths_from_listfile(args.source))
        else:
            p = Path(args.source)
            if p.is_file() or is_stream_like(args.source):
                total = None  # unknown for videos/streams; single file would be 1, but results may be multi-frame
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    total = 1
            elif p.is_dir():
                total = count_iter(iter_image_paths_from_dir(p))
            elif any(ch in args.source for ch in "*?[]"):
                total = count_iter(iter_image_paths_from_glob(args.source))
            else:
                total = None

    predict_name = "pred"

    # Run prediction as a generator (stream=True) → memory-safe
    results_gen = model.predict(
        source=src_for_ultralytics,   # dir/glob/list-file/list/single/stream
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        max_det=args.maxdet,
        stream=True,
        save=args.save_img,           # opt-in only
        project=str(outdir),
        name=predict_name,
        vid_stride=args.vid_stride,
        show=args.show,
        workers=args.workers,
        verbose=False
    )

    # Progress bar
    pbar = tqdm(total=total, unit="img", dynamic_ncols=True, miniters=1)
    processed = 0

    try:
        for res in results_gen:
            in_path = Path(res.path) if isinstance(res.path, (str, Path)) else Path(f"frame_{getattr(res, 'batch_idx', 0)}")

            # Save YOLO txt (with confidence) if requested
            if args.save_txt and getattr(res, "boxes", None) is not None and len(res.boxes) > 0:
                xywhn = res.boxes.xywhn.detach().cpu().numpy()
                clss  = res.boxes.cls.detach().cpu().numpy().astype(int)
                confs = res.boxes.conf.detach().cpu().numpy()
                lbl_out = labels_dir / f"{in_path.stem}.txt"
                with open(lbl_out, "w") as f:
                    for (cx, cy, w, h), c, conf in zip(xywhn, clss, confs):
                        f.write(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {conf:.6f}\n")

            # Save NDJSON if requested
            if json_fp is not None and getattr(res, "boxes", None) is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.detach().cpu().numpy()
                clss = res.boxes.cls.detach().cpu().numpy().astype(int)
                confs = res.boxes.conf.detach().cpu().numpy()
                H, W = int(res.orig_shape[0]), int(res.orig_shape[1])
                base = str(in_path)
                for (x1, y1, x2, y2), c, conf in zip(xyxy, clss, confs):
                    rec = {
                        "image": base,
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "conf": float(conf),
                        "class_id": int(c),
                        "class_name": names.get(int(c), str(int(c))),
                        "width": W,
                        "height": H
                    }
                    json_fp.write(json.dumps(rec) + "\n")

            processed += 1
            pbar.update(1)

            # Proactive GPU memory housekeeping for very long runs
            if torch.cuda.is_available() and (processed % 256 == 0):
                torch.cuda.empty_cache()

            # Drop strong references ASAP
            del res

    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
    finally:
        pbar.close()
        if json_fp is not None:
            json_fp.close()

    save_dir = outdir / predict_name
    print(f"\n✅ Done. Outputs:")
    print(f"  • Annotated media: {save_dir.resolve()}")
    if args.save_txt:
        print(f"  • YOLO labels:     {labels_dir.resolve()}")
    if args.save_json:
        print(f"  • NDJSON:          {Path(args.save_json).resolve()}")

if __name__ == "__main__":
    main()

