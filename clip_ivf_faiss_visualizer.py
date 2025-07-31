import os
import torch
import clip
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from sklearn.preprocessing import normalize
import base64
from io import BytesIO
import subprocess
import faiss

# ==== CONFIG ====
os.environ["OMP_NUM_THREADS"] = "1"  # Prevent OpenMP FAISS crashes
IMAGE_DIR = "designs"
OUTPUT_HTML = "clip_ivf_visualizer.html"
TOP_K = 5
SIMILARITY_THRESHOLD = 0.85
NLIST = 50  # Number of clusters
# ==== DEVICE SETUP ====
if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")

model, preprocess = clip.load("ViT-B/32", device=device)

def get_clip_embedding(img_path):
    try:
        image = Image.open(img_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model.encode_image(image_input)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.detach().cpu().numpy().copy().astype("float32")
    except Exception as e:
        print(f"❌ Failed to embed {img_path}: {e}")
        return None

def image_to_base64(img_path):
    with Image.open(img_path).convert("RGB") as img:
        img.thumbnail((128, 128))
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode()

# ==== EMBEDDINGS ====
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

# ==== FAISS INDEXING ====
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

# ==== GENERATE HTML ====
print("🌐 Generating HTML visualizer...")
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
        query_path = valid_paths[query_idx]
        f.write("<div class='query-section'>")
        f.write(f"<div class='row-title'>Query Image: {query_path.name}</div>")
        f.write("<div class='image-grid'>")

        # Query image
        f.write("<div class='img-card'>")
        f.write(f"<img src='data:image/jpeg;base64,{image_to_base64(query_path)}'><br>")
        f.write(f"<strong>{query_path.name}</strong><br><div class='sim-score'>Similarity: —</div>")
        f.write("</div>")

        for rank, idx in enumerate(neighbors[1:], start=1):
            if idx >= len(valid_paths):
                continue
            similarity = D[query_idx][rank]
            if similarity < SIMILARITY_THRESHOLD:
                continue
            similar_path = valid_paths[idx]
            similarity_pct = f"{similarity * 100:.1f}%"
            f.write("<div class='img-card'>")
            f.write(f"<img src='data:image/jpeg;base64,{image_to_base64(similar_path)}'><br>")
            f.write(f"{similar_path.name}<br><div class='sim-score'>Similarity: {similarity_pct}</div>")
            f.write("</div>")

        f.write("</div></div>")

    f.write("</body></html>")

# ==== OPEN HTML ON MAC ====
html_path = os.path.abspath(OUTPUT_HTML)
print(f"✅ Done. Opening in browser: {html_path}")
subprocess.run(["open", html_path])
