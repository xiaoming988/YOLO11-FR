from pathlib import Path
import shutil
import random
import csv

# ============================================================
# 1. Path settings
# ============================================================

src_root = Path(r"D:\yolo\data\YOLO11-FR\GYU-DET")
dst_root = Path(r"D:\yolo\data\YOLO11-FR\GYU-DET-Crack")

splits = ["train", "valid", "test"]

# In the original GYU-DET dataset, the class ID of Crack is 0
CRACK_ID = 0

# Whether to keep images without cracks as hard negative samples
KEEP_NEGATIVE_IMAGES = True

# Negative sample ratio control:
# None means keeping all non-crack negative samples
# 1.0 means the number of negative samples is at most equal to the number of positive samples
# 0.5 means the number of negative samples is at most half of the number of positive samples
NEGATIVE_RATIO = 0.5

# Whether to overwrite the output directory if it already exists
# It is recommended to set this to False for the first run to avoid accidentally deleting processed data
# If you want to regenerate the dataset, set this to True
OVERWRITE = True

# Supported image formats
image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ============================================================
# 2. Utility functions
# ============================================================

def prepare_output_dir():
    if dst_root.exists():
        if OVERWRITE:
            print(f"Output directory already exists. It will be deleted and regenerated: {dst_root}")
            shutil.rmtree(dst_root)
        else:
            raise FileExistsError(
                f"Output directory already exists: {dst_root}\n"
                f"If you are sure you want to regenerate the dataset, "
                f"change OVERWRITE = False to OVERWRITE = True in the script."
            )

    for split in splits:
        (dst_root / split / "images").mkdir(parents=True, exist_ok=True)
        (dst_root / split / "labels").mkdir(parents=True, exist_ok=True)


def read_yolo_label(label_path: Path):
    """
    Read YOLO bbox labels.
    Return crack_lines and total_boxes.
    crack_lines only contains the Crack class, and the class ID is rewritten as 0.
    """
    crack_lines = []
    total_boxes = 0

    if not label_path.exists():
        return crack_lines, total_boxes

    with open(label_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()

            # A YOLO bbox label should contain at least:
            # class x_center y_center width height
            if len(parts) < 5:
                continue

            try:
                cls_id = int(float(parts[0]))
            except ValueError:
                continue

            total_boxes += 1

            if cls_id == CRACK_ID:
                # Convert to single-class detection and rewrite the class ID as 0
                parts[0] = "0"

                # Keep only the first 5 columns of the bbox label
                # This is correct for standard YOLO detection labels
                crack_lines.append(" ".join(parts[:5]))

    return crack_lines, total_boxes


def collect_split_items(split: str):
    src_img_dir = src_root / split / "images"
    src_lbl_dir = src_root / split / "labels"

    if not src_img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {src_img_dir}")
    if not src_lbl_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {src_lbl_dir}")

    images = sorted([p for p in src_img_dir.iterdir() if p.suffix.lower() in image_exts])

    positive_items = []
    negative_items = []

    total_original_boxes = 0
    total_crack_boxes = 0
    missing_label_count = 0

    for img_path in images:
        label_path = src_lbl_dir / f"{img_path.stem}.txt"

        if not label_path.exists():
            missing_label_count += 1

        crack_lines, original_box_count = read_yolo_label(label_path)

        total_original_boxes += original_box_count
        total_crack_boxes += len(crack_lines)

        item = {
            "image_path": img_path,
            "crack_lines": crack_lines,
            "original_box_count": original_box_count,
            "crack_box_count": len(crack_lines),
        }

        if len(crack_lines) > 0:
            positive_items.append(item)
        else:
            negative_items.append(item)

    return {
        "split": split,
        "images": images,
        "positive_items": positive_items,
        "negative_items": negative_items,
        "total_original_boxes": total_original_boxes,
        "total_crack_boxes": total_crack_boxes,
        "missing_label_count": missing_label_count,
    }


def select_items(positive_items, negative_items):
    if not KEEP_NEGATIVE_IMAGES:
        return positive_items

    selected_negative_items = negative_items

    if NEGATIVE_RATIO is not None:
        max_negative = int(len(positive_items) * NEGATIVE_RATIO)
        random.seed(42)
        selected_negative_items = negative_items.copy()
        random.shuffle(selected_negative_items)
        selected_negative_items = selected_negative_items[:max_negative]

    return positive_items + selected_negative_items


def copy_items_to_output(split: str, selected_items):
    dst_img_dir = dst_root / split / "images"
    dst_lbl_dir = dst_root / split / "labels"

    for item in selected_items:
        img_path = item["image_path"]
        crack_lines = item["crack_lines"]

        dst_img_path = dst_img_dir / img_path.name
        dst_lbl_path = dst_lbl_dir / f"{img_path.stem}.txt"

        shutil.copy2(img_path, dst_img_path)

        # Create an empty txt file even if there is no crack.
        # This indicates that the image is a negative sample.
        with open(dst_lbl_path, "w", encoding="utf-8") as f:
            if crack_lines:
                f.write("\n".join(crack_lines) + "\n")


def write_classes_and_yaml():
    # Single-class classes.txt
    with open(dst_root / "classes.txt", "w", encoding="utf-8") as f:
        f.write("Crack\n")

    # YOLO dataset configuration file
    yaml_text = f"""path: {dst_root.as_posix()}
train: train/images
val: valid/images
test: test/images

names:
  0: Crack
"""

    with open(dst_root / "gyu_crack.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_text)


def write_summary(summary_rows):
    summary_csv = dst_root / "dataset_summary.csv"

    fieldnames = [
        "split",
        "original_images",
        "original_boxes",
        "crack_positive_images",
        "negative_images_available",
        "negative_images_kept",
        "output_images",
        "crack_boxes_kept",
        "missing_label_files",
    ]

    with open(summary_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    summary_txt = dst_root / "dataset_summary.txt"
    with open(summary_txt, "w", encoding="utf-8") as f:
        for row in summary_rows:
            f.write(f"[{row['split']}]\n")
            f.write(f"Original images: {row['original_images']}\n")
            f.write(f"Original annotation boxes: {row['original_boxes']}\n")
            f.write(f"Crack-positive images: {row['crack_positive_images']}\n")
            f.write(f"Available negative images: {row['negative_images_available']}\n")
            f.write(f"Kept negative images: {row['negative_images_kept']}\n")
            f.write(f"Output images: {row['output_images']}\n")
            f.write(f"Kept crack boxes: {row['crack_boxes_kept']}\n")
            f.write(f"Missing label files: {row['missing_label_files']}\n")
            f.write("\n")


def main():
    print("Start processing the GYU-DET single-class Crack dataset...")
    print(f"Source dataset path: {src_root}")
    print(f"Output dataset path: {dst_root}")
    print()

    prepare_output_dir()

    summary_rows = []

    for split in splits:
        info = collect_split_items(split)

        positive_items = info["positive_items"]
        negative_items = info["negative_items"]

        selected_items = select_items(positive_items, negative_items)

        copy_items_to_output(split, selected_items)

        negative_kept = len(selected_items) - len(positive_items)

        row = {
            "split": split,
            "original_images": len(info["images"]),
            "original_boxes": info["total_original_boxes"],
            "crack_positive_images": len(positive_items),
            "negative_images_available": len(negative_items),
            "negative_images_kept": negative_kept,
            "output_images": len(selected_items),
            "crack_boxes_kept": info["total_crack_boxes"],
            "missing_label_files": info["missing_label_count"],
        }

        summary_rows.append(row)

        print(f"[{split}]")
        print(f"  Original images: {row['original_images']}")
        print(f"  Original annotation boxes: {row['original_boxes']}")
        print(f"  Crack-positive images: {row['crack_positive_images']}")
        print(f"  Available negative images: {row['negative_images_available']}")
        print(f"  Kept negative images: {row['negative_images_kept']}")
        print(f"  Output images: {row['output_images']}")
        print(f"  Kept crack boxes: {row['crack_boxes_kept']}")
        print(f"  Missing label files: {row['missing_label_files']}")
        print()

    write_classes_and_yaml()
    write_summary(summary_rows)

    print("Processing completed.")
    print(f"New dataset path: {dst_root}")
    print(f"YOLO configuration file: {dst_root / 'gyu_crack.yaml'}")
    print(f"Summary text file: {dst_root / 'dataset_summary.txt'}")
    print(f"Summary CSV file: {dst_root / 'dataset_summary.csv'}")


if __name__ == "__main__":
    main()