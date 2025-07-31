import os
import csv
import torch
import clip
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ==== CONFIG ====
IMAGE_DIR = "/Users/sandeeps/Documents/PyImageSearch/designs"
SIMILARITY_THRESHOLD = 0.85
CSV_OUTPUT = "similar_images.csv"
TOP_K = 5  # Export top K similar pairs even if below threshold

# ==== DEVICE SETUP ====
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# ==== LOAD CLIP MODEL ====
model, preprocess = clip.load("ViT-B/32", device=device)

# ==== AUGMENTED EMBEDDING ====
def get_augmented_embedding(img_path):
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"❌ Could not open image: {img_path} — {e}")
        return None

    transforms = [
        lambda x: x,
        lambda x: x.rotate(90, expand=True),
        lambda x: x.rotate(180, expand=True),
        lambda x: x.transpose(Image.FLIP_LEFT_RIGHT)
    ]

    embeddings = []
    for t in transforms:
        aug = preprocess(t(img)).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model.encode_image(aug)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb.cpu().numpy())
    return np.mean(embeddings, axis=0).flatten()

# ==== STEP 1: Generate Embeddings ====
def compute_clip_embeddings(image_dir):
    embeddings = {}
    supported = ('.jpg', '.jpeg', '.pjpeg', '.png', '.bmp', '.webp')
    print(f"📁 Scanning directory: {image_dir}")

    for fname in sorted(os.listdir(image_dir)):
        if not fname.lower().endswith(supported):
            print(f"⛔ Skipping unsupported file: {fname}")
            continue

        fpath = os.path.join(image_dir, fname)
        emb = get_augmented_embedding(fpath)
        if emb is not None:
            embeddings[fname] = emb
            print(f"✅ Embedded: {fname}")
    return embeddings

# ==== STEP 2: Compare and Save ====
def find_and_export_similar_images(embeddings, threshold, output_csv, top_k=5):
    filenames = list(embeddings.keys())
    if len(filenames) < 2:
        print("❌ Not enough images to compare.")
        return

    vectors = np.array([embeddings[f] for f in filenames])
    sim_matrix = cosine_similarity(vectors)

    pairs = []
    for i in range(len(filenames)):
        for j in range(i + 1, len(filenames)):
            score = sim_matrix[i][j]
            pairs.append((filenames[i], filenames[j], score))

    pairs.sort(key=lambda x: x[2], reverse=True)

    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image 1 Name", "Image 2 Name", "Similarity Score"])

        count = 0
        for img1, img2, score in pairs:
            print(f"Similarity: {img1} ↔ {img2} = {score:.4f}")
            if score >= threshold or count < top_k:
                writer.writerow([img1, img2, round(score, 4)])
                count += 1

    print(f"\n✅ {count} similar image pairs written to: {output_csv}")

# ==== MAIN ====
if __name__ == "__main__":
    print("🔍 Generating augmented embeddings...")
    embs = compute_clip_embeddings(IMAGE_DIR)

    print("📊 Total embeddings created:", len(embs))
    find_and_export_similar_images(embs, SIMILARITY_THRESHOLD, CSV_OUTPUT, TOP_K)
