import os
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

def convert_all_images_to_jpg(source_dir, target_dir):
    os.makedirs(target_dir, exist_ok=True)

    all_image_paths = []
    subfolder_counts = defaultdict(int)

    print(f"\n🔍 Scanning folders inside: {source_dir}\n")

    for root, _, files in os.walk(source_dir):
        if not files:
            continue
        relative_path = os.path.relpath(root, source_dir)
        subfolder_label = relative_path if relative_path != '.' else "(root folder)"

        for file in files:
            file_path = os.path.join(root, file)
            all_image_paths.append(file_path)
            subfolder_counts[subfolder_label] += 1

    # Log subfolder-wise counts
    for folder, count in subfolder_counts.items():
        print(f"📂 {folder} → {count} image(s)")

    total_files = len(all_image_paths)
    print(f"\n📦 Total image files found: {total_files}\n")

    counter = 1
    converted_count = 0
    failed_files = []

    for file_path in tqdm(all_image_paths, desc="Converting", unit="img"):
        try:
            with Image.open(file_path) as img:
                img = img.convert("RGB")
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                target_file = os.path.join(target_dir, f"{base_name}.jpg")

                while os.path.exists(target_file):
                    target_file = os.path.join(target_dir, f"{base_name}_{counter}.jpg")
                    counter += 1

                img.save(target_file, "JPEG", quality=95)
                converted_count += 1
        except Exception as e:
            failed_files.append(f"{file_path} - {str(e)}")

    print("\n📊 Summary:")
    print(f"Total image files found:   {total_files}")
    print(f"Successfully converted:    {converted_count}")
    print(f"Failed during conversion:  {len(failed_files)}")

    if failed_files:
        print("\n❌ Failed Files:")
        for f in failed_files:
            print(" -", f)



# Example usage
source_folder = "/Users/sandeeps/Documents/PyImageSearch/DesignImage"
target_folder = "/Users/sandeeps/Documents/PyImageSearch/FinalDesignImages"
convert_all_images_to_jpg(source_folder, target_folder)
