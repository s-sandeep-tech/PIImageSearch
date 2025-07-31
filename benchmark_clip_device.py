import time
import torch
import clip
from PIL import Image

# ==== CONFIG ====
IMAGE_PATH = "/Users/sandeeps/Documents/PyImageSearch/images/image_003.jpg"  # ✅ Use a real image
REPEATS = 10  # How many times to run for timing

# ==== Load Image ====
img = Image.open(IMAGE_PATH).convert("RGB")

# ==== Benchmark Function ====
def benchmark(device_str):
    print(f"\n🧪 Benchmarking on {device_str.upper()}...")
    device = torch.device(device_str)

    model, preprocess = clip.load("ViT-B/32", device=device)
    img_tensor = preprocess(img).unsqueeze(0).to(device)

    # Warm-up
    with torch.no_grad():
        _ = model.encode_image(img_tensor)

    # Timed runs
    start = time.time()
    for _ in range(REPEATS):
        with torch.no_grad():
            _ = model.encode_image(img_tensor)
    end = time.time()

    total = end - start
    avg = total / REPEATS
    print(f"✅ Total time: {total:.4f}s for {REPEATS} runs")
    print(f"⏱️  Average per run: {avg:.4f}s")

# ==== Run Benchmarks ====
if torch.backends.mps.is_available():
    benchmark("mps")

benchmark("cpu")
