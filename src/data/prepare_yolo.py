
from pathlib import Path
import xml.etree.ElementTree as ET
import shutil
import yaml
from PIL import Image

# --------------------------------------------------
# PATHS AND CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = PROJECT_ROOT / "datasets" / "NEU-DET"

OUTPUT_ROOT = PROJECT_ROOT / "datasets" / "YOLO-NEU-DET-FINAL"

CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

CLASS_TO_ID = {
    name: index for index, name in enumerate(CLASSES)
}

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp"
}


# --------------------------------------------------
# CONVERT ONE DATASET SPLIT
# --------------------------------------------------

def convert_split(split):

    image_root = DATASET_ROOT / split / "images"
    annotation_root = DATASET_ROOT / split / "annotations"

    output_split = "train" if split == "train" else "val"

    output_images = OUTPUT_ROOT / "images" / output_split
    output_labels = OUTPUT_ROOT / "labels" / output_split

    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    # Index images using filename WITHOUT extension
    image_lookup = {}

    for image_path in image_root.rglob("*"):

        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        key = image_path.stem.lower()

        if key in image_lookup:
            raise ValueError(
                f"Duplicate image filename: {image_path.name}"
            )

        image_lookup[key] = image_path

    xml_files = sorted(annotation_root.glob("*.xml"))

    converted = 0
    total_objects = 0
    multi_object_images = 0
    skipped = 0

    for xml_path in xml_files:

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

        except ET.ParseError:
            print(f"Invalid XML: {xml_path.name}")
            skipped += 1
            continue

        filename = root.findtext("filename")

        if not filename:
            print(f"Missing filename: {xml_path.name}")
            skipped += 1
            continue

        # Handles filenames such as patches_104
        # and patches_104.jpg
        image_key = Path(filename).stem.lower()

        image_path = image_lookup.get(image_key)

        if image_path is None:
            print(f"Missing image: {filename}")
            skipped += 1
            continue

        with Image.open(image_path) as image:
            width, height = image.size

        if width <= 0 or height <= 0:
            skipped += 1
            continue

        yolo_lines = []

        for obj in root.findall("object"):

            class_name = obj.findtext("name")

            if class_name not in CLASS_TO_ID:
                continue

            box = obj.find("bndbox")

            if box is None:
                continue

            try:
                xmin = float(box.findtext("xmin"))
                ymin = float(box.findtext("ymin"))
                xmax = float(box.findtext("xmax"))
                ymax = float(box.findtext("ymax"))

            except (TypeError, ValueError):
                continue

            # Clamp coordinates to image boundaries
            xmin = max(0, min(xmin, width))
            ymin = max(0, min(ymin, height))
            xmax = max(0, min(xmax, width))
            ymax = max(0, min(ymax, height))

            if xmax <= xmin or ymax <= ymin:
                continue

            # Pascal VOC -> YOLO normalized coordinates
            x_center = ((xmin + xmax) / 2) / width
            y_center = ((ymin + ymax) / 2) / height

            box_width = (xmax - xmin) / width
            box_height = (ymax - ymin) / height

            class_id = CLASS_TO_ID[class_name]

            yolo_lines.append(
                f"{class_id} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{box_width:.6f} "
                f"{box_height:.6f}"
            )

        if not yolo_lines:
            print(f"No valid boxes: {xml_path.name}")
            skipped += 1
            continue

        destination_image = output_images / image_path.name

        destination_label = (
            output_labels / f"{image_path.stem}.txt"
        )

        shutil.copy2(image_path, destination_image)

        destination_label.write_text(
            "\n".join(yolo_lines) + "\n",
            encoding="utf-8"
        )

        converted += 1
        total_objects += len(yolo_lines)

        if len(yolo_lines) > 1:
            multi_object_images += 1

    print(f"\n========== {split.upper()} RESULTS ==========")
    print("Images converted:", converted)
    print("Defect objects:", total_objects)
    print("Images with multiple defects:", multi_object_images)
    print("Skipped:", skipped)

    return converted, total_objects, skipped


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_ROOT}"
        )

    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        raise FileExistsError(
            f"Output folder already contains files: {OUTPUT_ROOT}"
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    train_images, train_objects, train_skipped = (
        convert_split("train")
    )

    val_images, val_objects, val_skipped = (
        convert_split("validation")
    )

    # Generate YOLO configuration
    data_config = {
        "path": OUTPUT_ROOT.resolve().as_posix(),
        "train": "images/train",
        "val": "images/val",
        "nc": len(CLASSES),
        "names": CLASSES,
    }

    with open(OUTPUT_ROOT / "data.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(
            data_config,
            file,
            sort_keys=False
        )

    # Final report
    print("\n========== YOLO DATASET CREATED ==========")

    print("Output:", OUTPUT_ROOT)

    print("Training images:", train_images)
    print("Validation images:", val_images)

    print("Total images:", train_images + val_images)

    print("Training objects:", train_objects)
    print("Validation objects:", val_objects)

    print("Total defect objects:", train_objects + val_objects)

    print("Total skipped:", train_skipped + val_skipped)

    print("Configuration:", OUTPUT_ROOT / "data.yaml")

    if train_images != 1440 or val_images != 360:
        print("\nWARNING: Converted image counts differ from expected counts.")

    else:
        print("\nSUCCESS: All 1800 expected images converted!")


if __name__ == "__main__":
    main()

