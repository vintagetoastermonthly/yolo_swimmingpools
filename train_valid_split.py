import random
from pathlib import Path

root = Path("datasets/SwimmingPools_Auckland_090925")
base_txt = root / "files.txt"
train_txt = root / "train.txt"
val_txt   = root / "val.txt"

# Read all image paths from train.txt
with open(base_txt, "r") as f:
    images = [line.strip() for line in f if line.strip()]

random.seed(42)  # reproducibility
random.shuffle(images)

# 80/20 split
split_idx = int(len(images) * 0.9)
train_split = images[:split_idx]
val_split   = images[split_idx:]

# Write them out
with open(train_txt, "w") as f:
    f.write("\n".join(train_split) + "\n")

with open(val_txt, "w") as f:
    f.write("\n".join(val_split) + "\n")

print(f"Train: {len(train_split)} images")
print(f"Val:   {len(val_split)} images")

