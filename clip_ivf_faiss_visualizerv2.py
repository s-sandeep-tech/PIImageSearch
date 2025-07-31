import os
import torch
import clip
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from sklearn.preprocessing import normalize
import subprocess
import faiss
import pickle

# ==== CONFIG ====
os.environ["OMP_NUM_THREADS"] = "1"
IMAGE_DIR = "FinalDesignImages"
OUTPUT_HTML = "clip_ivf_visualizer.html"
INDEX_PATH = "clip_ivf.index"
PATHS_PKL = "clip_image_paths.pkl"
TOP_K = 5
SIMILARITY_THRESHOLD = 0.96
NLIST = 50
RESTORE_MODE = True  # Set to True to load index and skip recomputing

# ==== DEVICE SETUP ====
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")
model, preprocess = clip.load("ViT-B/32", device=device)

def get_clip_embedding(img_path):
    try:
        image = Image.open(img_path)
        if image.mode in ["P", "LA"]:
            image = image.convert("RGBA")
        else:
            image = image.convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model.encode_image(image_input)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.detach().cpu().numpy().astype("float32")
    except Exception as e:
        print(f"❌ Failed to embed {img_path}: {e}")
        return None

# ==== RESTORE OR COMPUTE EMBEDDINGS ====
if RESTORE_MODE and os.path.exists(INDEX_PATH) and os.path.exists(PATHS_PKL):
    print("📥 Loading FAISS index and image paths...")
    index = faiss.read_index(INDEX_PATH)
    with open(PATHS_PKL, "rb") as f:
        valid_paths = pickle.load(f)
    X = None  # not needed for search if just using saved index
    print(f"✅ Restored {len(valid_paths)} image paths from disk.")
    print(f"✅ FAISS index loaded from: {INDEX_PATH}")
else:
    print("🔄 Computing embeddings...")
    image_paths = list(Path(IMAGE_DIR).glob("*.jpg")) + list(Path(IMAGE_DIR).glob("*.png"))
    embeddings = []
    valid_paths = []

    for path in tqdm(image_paths):
        emb = get_clip_embedding(str(path))
        if emb is not None:
            embeddings.append(emb)
            valid_paths.append(path)

    print(f"📦 {len(embeddings)} embeddings computed.")

    X = np.array(embeddings).astype("float32")
    if X.ndim == 3:
        X = X.squeeze()
    X = normalize(X)
    dim = X.shape[1]

    print("📡 Preparing FAISS index...")
    try:
        if len(X) < NLIST * 39:
            raise ValueError("Too few samples for IVF. Using FlatIP.")

        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, NLIST, faiss.METRIC_INNER_PRODUCT)

        print("🔧 Training index...")
        index.train(X)
        index.add(X)
        print("🔍 Searching with IVF...")
        D, I = index.search(X, TOP_K + 1)
    except Exception as e:
        print(f"⚠️ {e} — fallback to FlatIP...")
        index = faiss.IndexFlatIP(dim)
        index.add(X)
        print("🔍 Searching with FlatIP...")
        D, I = index.search(X, TOP_K + 1)

    faiss.write_index(index, INDEX_PATH)
    with open(PATHS_PKL, "wb") as f:
        pickle.dump(valid_paths, f)
    print(f"💾 FAISS index and image paths saved to disk.")

# ==== HTML GENERATOR ====
def generate_similarity_html(valid_paths, I, D, OUTPUT_HTML, SIMILARITY_THRESHOLD):
    print("🌐 Generating HTML visualizer...")
    skipped = 0

    with open(OUTPUT_HTML, "w") as f:
        f.write("""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Image Similarity Viewer</title>
    <style>
        body { font-family: 'Segoe UI', sans-serif; background: #f5f5f5; padding: 20px; }
        h1 { text-align: center; color: #333; }
        .query-section { margin-bottom: 40px; padding: 20px; background: white; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
        .image-grid { display: flex; flex-wrap: wrap; gap: 20px; }
        .img-card { background: #fff; border-radius: 10px; padding: 10px; text-align: center; width: 160px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); transition: transform 0.2s ease; }
        .img-card:hover { transform: scale(1.03); }
        .img-card img { max-width: 128px; max-height: 128px; border-radius: 6px; border: 1px solid #ccc; }
        .sim-score { font-size: 0.9em; color: #666; margin-top: 4px; }
        .row-title { margin-bottom: 16px; font-size: 18px; font-weight: 600; color: #444; }
    </style>
</head>
<body>
    <h1>Image Similarity Viewer</h1>
""")

        for query_idx, neighbors in enumerate(I):
            valid_matches = [
                (idx, D[query_idx][rank])
                for rank, idx in enumerate(neighbors[1:], start=1)
                if (
                    idx < len(valid_paths) and
                    D[query_idx][rank] >= SIMILARITY_THRESHOLD and
                    valid_paths[idx].name != valid_paths[query_idx].name
                )
            ]

            print(f"🔍 {valid_paths[query_idx].name} → {len(valid_matches)} valid matches above {SIMILARITY_THRESHOLD * 100:.1f}%")

            if not valid_matches:
                skipped += 1
                continue

            query_path = valid_paths[query_idx]
            f.write("<div class='query-section'>")
            f.write(f"<div class='row-title'>Query Image: {query_path.name}</div>")
            f.write("<div class='image-grid'>")

            # Query image
            f.write("<div class='img-card'>")
            f.write(f"<img src='{IMAGE_DIR}/{query_path.name}'><br>")
            f.write(f"<strong>{query_path.name}</strong><br><div class='sim-score'>Similarity: —</div>")
            f.write("</div>")

            # Matched images
            for idx, similarity in valid_matches:
                similar_path = valid_paths[idx]
                similarity_pct = f"{similarity * 100:.1f}%"
                f.write("<div class='img-card'>")
                f.write(f"<img src='{IMAGE_DIR}/{similar_path.name}'><br>")
                f.write(f"{similar_path.name}<br><div class='sim-score'>Similarity: {similarity_pct}</div>")
                f.write("</div>")

            f.write("</div></div>")

        f.write("</body></html>")

    print(f"✅ HTML written to: {OUTPUT_HTML}")
    if skipped > 0:
        print(f"ℹ️ Skipped {skipped} query images with no valid matches above threshold.")

# ==== RUN VIEWER ====
if not RESTORE_MODE:
    generate_similarity_html(valid_paths, I, D, OUTPUT_HTML, SIMILARITY_THRESHOLD)

    html_path = os.path.abspath(OUTPUT_HTML)
    print(f"✅ Done. Opening in browser: {html_path}")
    subprocess.run(["open", html_path])
else:
    print("🛑 Skipped HTML generation in restore mode (no distances loaded).")
