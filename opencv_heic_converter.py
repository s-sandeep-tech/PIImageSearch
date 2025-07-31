
import os
from pathlib import Path
import cv2
from PIL import Image
import pyheif

def convert_with_opencv(input_path, output_path):
    try:
        img = cv2.imread(str(input_path))
        if img is not None:
            cv2.imwrite(str(output_path), img)
            return True
        else:
            print(f"⚠️ OpenCV failed to read {input_path.name}")
            return False
    except Exception as e:
        print(f"❌ OpenCV error on {input_path.name}: {e}")
        return False

def convert_heic_to_jpeg(input_path, output_path):
    try:
        heif_file = pyheif.read(input_path)
        image = Image.frombytes(
            heif_file.mode,
            heif_file.size,
            heif_file.data,
            "raw",
            heif_file.mode,
            heif_file.stride,
        )
        image.convert("RGB").save(output_path, "JPEG", quality=95)
        return True
    except Exception as e:
        print(f"❌ HEIC error on {input_path.name}: {e}")
        return False

def bulk_convert(input_folder="originals", output_folder="designs"):
    input_dir = Path(input_folder)
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    supported_exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".heic"]
    converted = 0

    for file in input_dir.iterdir():
        if file.suffix.lower() in supported_exts:
            out_file = output_dir / (file.stem + ".jpg")

            if file.suffix.lower() == ".heic":
                success = convert_heic_to_jpeg(file, out_file)
            else:
                success = convert_with_opencv(file, out_file)

            if success:
                print(f"✅ Converted: {file.name} → {out_file.name}")
                converted += 1

    print(f"🎉 Finished! Converted {converted} images to JPEG.")

if __name__ == "__main__":
    bulk_convert("images", "designs")
