import os
import csv
import argparse
import random

def create_metadata_csv(data_dir, output_csv):
    """
    Scans data_dir for images and writes a CSV with random labels (0..4).
    """
    image_files = []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".bmp")):
                rel_path = os.path.relpath(os.path.join(root, f), data_dir)
                image_files.append(rel_path)

    with open(output_csv, "w", newline="") as f:
        fieldnames = ["filename", "label"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for img in image_files:
            label = random.randint(0, 4)  # e.g., 5 classes
            writer.writerow({"filename": img, "label": label})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Directory containing images")
    parser.add_argument("--output_csv", required=True, help="Where to save the metadata CSV")
    args = parser.parse_args()

    create_metadata_csv(args.data_dir, args.output_csv)

if __name__ == "__main__":
    main()
