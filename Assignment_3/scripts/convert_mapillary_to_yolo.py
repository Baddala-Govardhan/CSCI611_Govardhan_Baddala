import json
import random
import shutil
from pathlib import Path
from PIL import Image

RAW = Path("data/mapillary_raw")
OUT_IMG = Path("data/images")
OUT_LBL = Path("data/labels")

random.seed(42)

# collect all images
images = {}
for img in RAW.rglob("*.jpg"):
    images[img.stem] = img

# collect all json
annotations = {}
for js in RAW.rglob("*.json"):
    annotations[js.stem] = js

# match pairs
pairs = []
for stem in images:
    if stem in annotations:
        pairs.append((images[stem], annotations[stem]))

print("Total matched:", len(pairs))

# take small subset
pairs = pairs[:1000]

# split
train = pairs[:800]
val = pairs[800:900]
test = pairs[900:1000]

print("Train:", len(train), "Val:", len(val), "Test:", len(test))

def save_split(data, split_name):
    for img_path, json_path in data:
        with Image.open(img_path) as img:
            w, h = img.size

        with open(json_path) as f:
            data = json.load(f)

        boxes = []
        for obj in data.get("objects", []):
            bbox = obj.get("bbox", {})
            if all(k in bbox for k in ["xmin", "ymin", "xmax", "ymax"]):
                xmin = bbox["xmin"]
                ymin = bbox["ymin"]
                xmax = bbox["xmax"]
                ymax = bbox["ymax"]

                bw = xmax - xmin
                bh = ymax - ymin
                xc = xmin + bw / 2
                yc = ymin + bh / 2

                boxes.append(
                    f"0 {xc/w:.6f} {yc/h:.6f} {bw/w:.6f} {bh/h:.6f}"
                )

        # copy image
        shutil.copy(img_path, OUT_IMG / split_name / img_path.name)

        # save label
        label_file = OUT_LBL / split_name / (img_path.stem + ".txt")
        label_file.write_text("\n".join(boxes))


for split_name, dataset in [
    ("train", train),
    ("val", val),
    ("test", test),
]:
    save_split(dataset, split_name)

print("Done converting.")