  python visualize_segmentations.py \
    --images ./datasets/SwimmingPools_Auckland_090925/images/train/data_for_cvat/Auckland_2024_Urban_sample/ \
    --labels segmentations_sam2 \
    --outdir visualizations/sam2_opaque \
    --sample 2000 \
    --alpha 0.6 \
    --class-names SwimmingPool PaddlingPool
