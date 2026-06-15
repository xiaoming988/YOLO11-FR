from pathlib import Path
from typing import Optional
import shutil
import cv2
import numpy as np


# =========================
# 1. Set dataset paths before running
# =========================
ROOT = Path(r"D:\yolo\data\YOLO11-FR\CRACK500")
OUT = Path(r"D:\yolo\data\YOLO11-FR\CRACK500_YOLO")


# =========================
# 2. Path resolution
# =========================
def resolve_path(rel_path: str) -> Optional[Path]:
   
    rel_path = rel_path.strip().replace("\\", "/")
    rel = Path(rel_path)

    # Case 1: standard path, e.g., ROOT/traincrop/xxx.jpg
    p1 = ROOT / rel
    if p1.exists():
        return p1

    # Case 2: nested split-folder path, e.g., ROOT/traincrop/traincrop/xxx.jpg
    if len(rel.parts) >= 2:
        p2 = ROOT / rel.parts[0] / rel
        if p2.exists():
            return p2

    # Case 3: fallback search by filename
    filename = rel.name
    candidates = list(ROOT.rglob(filename))
    if candidates:
        # Prefer paths that contain the split-folder name.
        split_folder = rel.parts[0].lower() if len(rel.parts) > 0 else ""
        for c in candidates:
            if split_folder in str(c).lower():
                return c
        return candidates[0]

    return None


# =========================
# 3. Read image-mask pairs
# =========================
def parse_pairs(split: str):
    txt_path = ROOT / f"{split}.txt"
    if not txt_path.exists():
        raise FileNotFoundError(f"Cannot find txt file: {txt_path}")

    pairs = []
    missing_count = 0

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line_id, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 2:
                print(f"[WARN] {split}.txt line {line_id} has fewer than 2 columns: {line}")
                continue

            img_path = resolve_path(parts[0])
            mask_path = resolve_path(parts[1])

            if img_path is not None and mask_path is not None:
                pairs.append((img_path, mask_path))
            else:
                missing_count += 1
                if missing_count <= 5:
                    print(f"[MISSING] {split} line {line_id}")
                    print(" image rel:", parts[0], "resolved:", img_path)
                    print(" mask  rel:", parts[1], "resolved:", mask_path)

    print(f"{split}: found {len(pairs)} pairs, missing {missing_count}")
    return pairs


# =========================
# 4. Convert masks to YOLO bounding boxes
# =========================
def mask_to_yolo_boxes(mask_path: Path, min_area=15, dilate_iter=2):
    """
    Convert a Crack500 binary mask into YOLO-format bounding boxes.

    Notes:
    - Non-black pixels in the mask are treated as crack regions.
    - Connected components are extracted and converted into bounding boxes.
    - dilate_iter applies slight dilation to reduce excessive fragmentation of thin cracks.
    """

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        print(f"[WARN] Cannot read mask: {mask_path}")
        return []

    h, w = mask.shape[:2]

    binary = (mask > 0).astype(np.uint8)

    if dilate_iter > 0:
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=dilate_iter)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    boxes = []

    for i in range(1, num_labels):
        x, y, bw, bh, area = stats[i]

        if area < min_area:
            continue

        x_center = (x + bw / 2) / w
        y_center = (y + bh / 2) / h
        box_w = bw / w
        box_h = bh / h

        x_center = min(max(x_center, 0.0), 1.0)
        y_center = min(max(y_center, 0.0), 1.0)
        box_w = min(max(box_w, 0.0), 1.0)
        box_h = min(max(box_h, 0.0), 1.0)

        boxes.append((0, x_center, y_center, box_w, box_h))

    return boxes


# =========================
# 5. Convert one split
# =========================
def convert_split(split: str):
    pairs = parse_pairs(split)

    image_out_dir = OUT / "images" / split
    label_out_dir = OUT / "labels" / split

    image_out_dir.mkdir(parents=True, exist_ok=True)
    label_out_dir.mkdir(parents=True, exist_ok=True)

    empty_labels = 0
    total_boxes = 0

    for idx, (img_path, mask_path) in enumerate(pairs):
        # Avoid filename conflicts caused by identical image names in different folders.
        dst_img_name = f"{split}_{idx:06d}_{img_path.name}"
        dst_img_path = image_out_dir / dst_img_name

        shutil.copy2(img_path, dst_img_path)

        boxes = mask_to_yolo_boxes(mask_path)

        label_path = label_out_dir / f"{Path(dst_img_name).stem}.txt"

        with open(label_path, "w", encoding="utf-8") as f:
            for cls, xc, yc, bw, bh in boxes:
                f.write(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

        if len(boxes) == 0:
            empty_labels += 1

        total_boxes += len(boxes)

    print(f"{split}: converted images = {len(pairs)}")
    print(f"{split}: empty labels = {empty_labels}")
    print(f"{split}: total boxes = {total_boxes}")


# =========================
# 6. Generate YOLO dataset yaml
# =========================
def write_yaml():
    yaml_text = f"""path: {OUT.as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: crack
"""

    yaml_path = OUT / "crack500.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_text)

    print(f"YAML saved to: {yaml_path}")


# =========================
# 7. Main program
# =========================
if __name__ == "__main__":
    print("ROOT:", ROOT)
    print("ROOT exists:", ROOT.exists())

    if not ROOT.exists():
        raise FileNotFoundError(f"ROOT does not exist: {ROOT}")

    # Remove old output to avoid mixing previous conversion results.
    if OUT.exists():
        print("Removing old output folder:", OUT)
        shutil.rmtree(OUT)

    for split_name in ["train", "val", "test"]:
        print(f"\nConverting {split_name}...")
        convert_split(split_name)

    write_yaml()

    print("\nDone.")
    print("YOLO dataset saved to:", OUT)