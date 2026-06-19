"""
Step 1: Convert XML annotations → YOLO format .txt files
Run this ONCE before training.
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil
import random

# ─── CONFIG ───────────────────────────────────────────────────────────────────
IMAGES_DIR      = "dataset/images"
ANNOTATIONS_DIR = "dataset/annotations"
OUTPUT_DIR      = "dataset/yolo"          # will be created automatically
TRAIN_RATIO     = 0.70
VAL_RATIO       = 0.15
# TEST = remaining 0.15
# ──────────────────────────────────────────────────────────────────────────────


def xml_to_yolo(xml_path, image_dir):
    """Parse one XML file and return YOLO-format lines."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find("filename").text
    size     = root.find("size")
    img_w    = int(size.find("width").text)
    img_h    = int(size.find("height").text)

    yolo_lines = []
    for obj in root.findall("object"):
        bndbox = obj.find("bndbox")
        xmin = int(float(bndbox.find("xmin").text))
        ymin = int(float(bndbox.find("ymin").text))
        xmax = int(float(bndbox.find("xmax").text))
        ymax = int(float(bndbox.find("ymax").text))

        # Convert to YOLO normalised cx, cy, w, h
        cx = ((xmin + xmax) / 2) / img_w
        cy = ((ymin + ymax) / 2) / img_h
        bw = (xmax - xmin)       / img_w
        bh = (ymax - ymin)       / img_h

        yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    return filename, yolo_lines


def main():
    images_path = Path(IMAGES_DIR)
    ann_path    = Path(ANNOTATIONS_DIR)
    out_path    = Path(OUTPUT_DIR)

    # Create output folder tree
    for split in ("train", "val", "test"):
        (out_path / split / "images").mkdir(parents=True, exist_ok=True)
        (out_path / split / "labels").mkdir(parents=True, exist_ok=True)

    # Collect all XML files
    xml_files = sorted(ann_path.glob("*.xml"))
    if not xml_files:
        print(f"[ERROR] No XML files found in {ann_path}")
        return

    print(f"[INFO] Found {len(xml_files)} annotation files")

    # Shuffle & split
    random.seed(42)
    random.shuffle(xml_files)
    n        = len(xml_files)
    n_train  = int(n * TRAIN_RATIO)
    n_val    = int(n * VAL_RATIO)

    splits = {
        "train": xml_files[:n_train],
        "val"  : xml_files[n_train : n_train + n_val],
        "test" : xml_files[n_train + n_val :],
    }

    for split, files in splits.items():
        print(f"[INFO] {split}: {len(files)} files")
        for xml_file in files:
            filename, yolo_lines = xml_to_yolo(xml_file, images_path)

            # Find matching image (try common extensions)
            img_src = None
            for ext in (".jpg", ".jpeg", ".png", ".bmp"):
                candidate = images_path / (Path(filename).stem + ext)
                if candidate.exists():
                    img_src = candidate
                    break
                # Also try exact filename
                candidate2 = images_path / filename
                if candidate2.exists():
                    img_src = candidate2
                    break

            if img_src is None:
                print(f"[WARN] Image not found for {xml_file.name}, skipping")
                continue

            # Copy image
            shutil.copy(img_src, out_path / split / "images" / img_src.name)

            # Write label
            label_name = img_src.stem + ".txt"
            label_path = out_path / split / "labels" / label_name
            with open(label_path, "w") as f:
                f.write("\n".join(yolo_lines))

    # Write dataset YAML for YOLOv8
    yaml_content = f"""\
# ANPDRS Dataset - YOLOv8 Config
path: {out_path.resolve().as_posix()}
train: train/images
val:   val/images
test:  test/images

nc: 1
names: ['license_plate']
"""
    yaml_path = out_path / "plates.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"\n[DONE] YOLO dataset ready at: {out_path}")
    print(f"[DONE] YAML config saved at : {yaml_path}")
    print("\nNext step: run  python train_yolo.py")


if __name__ == "__main__":
    main()