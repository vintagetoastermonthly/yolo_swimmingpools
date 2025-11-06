#!/bin/bash
# Complete End-to-End Pipeline: Training → Inference → Segmentation → Geospatial Export
# RT-DETR + SAM2 + GeoPackage (All Apache 2.0 Licensed)

set -e  # Exit on error

echo "========================================="
echo "Swimming Pool Detection Pipeline"
echo "========================================="
echo ""

# Configuration
MODEL_PATH="${1:-runs/train/rtdetr_hf/final_model}"
IMAGES_DIR="${2:-images}"
OUTPUT_BASE="${3:-pipeline_output}"
CONF_THRESHOLD="${4:-0.5}"
SAM_MODEL="${5:-facebook/sam2-hiera-tiny}"

# Create output directories
mkdir -p "$OUTPUT_BASE/detections"
mkdir -p "$OUTPUT_BASE/segmentations"

echo "Configuration:"
echo "  Model:       $MODEL_PATH"
echo "  Images:      $IMAGES_DIR"
echo "  Output:      $OUTPUT_BASE"
echo "  Confidence:  $CONF_THRESHOLD"
echo "  SAM Model:   $SAM_MODEL"
echo ""

# Step 1: RT-DETR Detection
echo "========================================="
echo "Step 1: RT-DETR Bounding Box Detection"
echo "========================================="
python infer_hf.py \
    --model "$MODEL_PATH" \
    --source "$IMAGES_DIR" \
    --outdir "$OUTPUT_BASE/detections" \
    --save-txt \
    --conf "$CONF_THRESHOLD"

echo ""
echo "✅ Detection complete!"
echo ""

# Step 2: SAM2 Segmentation
echo "========================================="
echo "Step 2: SAM2 Auto-Segmentation"
echo "========================================="
python autoseg_sam2.py \
    --images "$IMAGES_DIR" \
    --labels "$OUTPUT_BASE/detections/labels" \
    --outdir "$OUTPUT_BASE/segmentations" \
    --sam-model "$SAM_MODEL" \
    --conf "$CONF_THRESHOLD"

echo ""
echo "✅ Segmentation complete!"
echo ""

# Step 3: Geospatial Export
echo "========================================="
echo "Step 3: Export to GeoPackage"
echo "========================================="
python convert_yolo_seg_to_gpkg.py \
    --labels "$OUTPUT_BASE/segmentations" \
    --images "$IMAGES_DIR" \
    --output "$OUTPUT_BASE/pools_segmented.gpkg" \
    --overwrite

echo ""
echo "✅ Geospatial export complete!"
echo ""

# Summary
echo "========================================="
echo "Pipeline Complete!"
echo "========================================="
echo ""
echo "Outputs:"
echo "  Bounding boxes:  $OUTPUT_BASE/detections/labels/"
echo "  Segmentations:   $OUTPUT_BASE/segmentations/"
echo "  GeoPackage:      $OUTPUT_BASE/pools_segmented.gpkg"
echo ""
echo "Next steps:"
echo "  - Open GeoPackage in QGIS for visualization"
echo "  - Export to other GIS formats (Shapefile, GeoJSON, etc.)"
echo "  - Perform geospatial analysis"
echo ""
