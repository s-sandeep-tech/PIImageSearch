import os

# === CONFIG ===
FOLDER = "/Users/sandeeps/Documents/PyImageSearch/images"  # ✅ your image folder
PREFIX = "image_"
SUPPORTED = ('.jpg', '.jpeg', '.pjpeg', '.png', '.bmp', '.webp')

# === RENAME START ===
files = [f for f in os.listdir(FOLDER) if f.lower().endswith(SUPPORTED)]
files.sort()  # optional: sort by name

for i, old_name in enumerate(files, start=1):
    ext = os.path.splitext(old_name)[1].lower()
    new_name = f"{PREFIX}{i:03d}{ext}"
    old_path = os.path.join(FOLDER, old_name)
    new_path = os.path.join(FOLDER, new_name)

    os.rename(old_path, new_path)
    print(f"✅ Renamed: {old_name} → {new_name}")

print(f"\n🎉 Renamed {len(files)} files.")
