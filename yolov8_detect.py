import os
import csv
from pathlib import Path
from ultralytics import YOLO
import torch
import cv2

# ========= CONFIG =========
IMAGE_DIR   = Path("/Users/sandeeps/Documents/PyImageSearch/images")  # ⇦ your images
OUTPUT_DIR  = Path("yolo_outputs")          # annotated images will go here
CSV_PATH    = Path("detections.csv")        # set None to skip CSV
CONF_THRES  = 0.25                          # confidence threshold (0‑1)

# ========= SETUP =========
OUTPUT_DIR.mkdir(exist_ok=True)
# ==== DEVICE SETUP ====
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")
  # 'mps' for Apple Silicon

model = YOLO("yolov8n.pt")       # nano model (fast & small). use 'yolov8s.pt' for better accuracy

print(f"Running on device: {device}")
print("Model loaded:", model.names)

# ========= OPTIONAL CSV =========
csv_file = None
if CSV_PATH:
    csv_file = open(CSV_PATH, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["filename", "label", "confidence", "xmin", "ymin", "xmax", "ymax"])

# ========= PROCESS IMAGES =========
supported_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".pjpeg")
img_files = [p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in supported_exts]

for img_path in img_files:
    print(f"🔍 Detecting: {img_path.name}")
    # run inference
    results = model(img_path, conf=CONF_THRES, device=device, verbose=False)

    # YOLO returns list; grab first result
    res = results[0]

    # draw detections on image
    annotated = res.plot()  # numpy array BGR

    # save annotated image
    out_file = OUTPUT_DIR / img_path.name
    cv2.imwrite(str(out_file), annotated)
    print(f"✅ Saved: {out_file.name} ({len(res.boxes)} objects)")

    # write detections to CSV
    if csv_file:
        for box in res.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf  = float(box.conf[0])
            xmin, ymin, xmax, ymax = map(float, box.xyxy[0])
            writer.writerow([img_path.name, label, round(conf, 4),
                             round(xmin), round(ymin), round(xmax), round(ymax)])

# ========= CLEANUP =========
if csv_file:
    csv_file.close()
    print(f"\n📄 Detection CSV saved to: {CSV_PATH.resolve()}")

print("\n🎉 All done!")
