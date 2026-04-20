"""
Splits pokemon_raw/<class>/*.jpg into pokemon_dataset/{train,val}/<class>/
using an 80/20 split.
"""

import shutil
import random
from pathlib import Path

SRC = Path("pokemon_raw")
DST = Path("pokemon_dataset")
TRAIN_RATIO = 0.8
SEED = 42  # makes the split reproducible

random.seed(SEED)

# Wipe any previous run so we start clean
if DST.exists():
    shutil.rmtree(DST)

total_train = 0
total_val = 0

for class_dir in sorted(SRC.iterdir()):
    if not class_dir.is_dir():
        continue

    images = list(class_dir.glob("*.*"))
    random.shuffle(images)

    cut = int(len(images) * TRAIN_RATIO)
    train_files = images[:cut]
    val_files = images[cut:]

    train_out = DST / "train" / class_dir.name
    val_out = DST / "val" / class_dir.name
    train_out.mkdir(parents=True, exist_ok=True)
    val_out.mkdir(parents=True, exist_ok=True)

    for f in train_files:
        shutil.copy(f, train_out / f.name)
    for f in val_files:
        shutil.copy(f, val_out / f.name)

    total_train += len(train_files)
    total_val += len(val_files)
    print(f"{class_dir.name:15} train={len(train_files):3}  val={len(val_files):3}")

print(f"\nTOTAL  train={total_train}  val={total_val}")