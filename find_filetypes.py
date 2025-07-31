from pathlib import Path
from collections import Counter

def scan_image_types(folder="originals"):
    path = Path(folder)
    extensions = []

    for file in path.glob("*.*"):
        ext = file.suffix.lower()
        if ext:
            extensions.append(ext)

    count = Counter(extensions)

    print(f"\n📂 Found {len(extensions)} image files in '{folder}'")
    print("📊 Image Type Breakdown:\n")
    for ext, qty in count.items():
        print(f"  {ext}: {qty} files")

    return count

# Example usage
if __name__ == "__main__":
    scan_image_types("Images")
