# pip install ultralytics
# Note: Using RT-DETR (Apache 2.0 licensed) for detection + SAM2 for segmentation
from ultralytics.data.annotator import auto_annotate

auto_annotate(
    data="images/",                # folder with images
    det_model="weights/best.pt", # your trained RT-DETR detector
    sam_model="sam2_b.pt",                # SAM2 (or sam2.1_*). Also works with "sam_b.pt", "mobile_sam.pt"
    device="cuda",                        # or 'cpu'
    conf=0.25, iou=0.45, imgsz=640, max_det=300,
    output_dir="output"      # YOLO-seg polygons will be written here
)
