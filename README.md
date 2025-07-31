# CLIP + FAISS Image Similarity Viewer

This project computes **image embeddings using OpenAI's CLIP model**, indexes them with **FAISS**, and generates an **interactive HTML viewer** to visualize **similar images** along with metadata from an Excel sheet.

---

## 📌 Features

- ✅ Supports **JPEG, PNG** images (handles transparency automatically)  
- ✅ Uses **CLIP (ViT-B/32)** to generate embeddings  
- ✅ **FAISS Index** for fast similarity search (IVF or FlatIP fallback)  
- ✅ **Threshold-based filtering** of matches  
- ✅ **Sorted results** (highest similarity first)  
- ✅ **Dynamic HTML output** with:
  - Query image
  - Top-K most similar matches
  - Metadata from Excel (filtered fields)
  - Beautiful UI cards with hover effects
- ✅ Supports **FAISS index persistence** (`.index`, `.pkl`, `.npy`)  
- ✅ Supports **Excel metadata filtering** (skips unwanted fields)  
- ✅ Optional **PDF export** of the HTML report  

---

## 📂 Project Structure

project/
│── FinalDesignImages/          # Folder containing all images
│── design_metadata.xlsx        # Excel file with image metadata
│── clip_ivf_visualizer.html    # Generated HTML file
│── clip_ivf_visualizer.pdf     # (Optional) Generated PDF file
│── clip_ivf.index              # Saved FAISS index
│── clip_image_paths.pkl        # Saved image paths
│── clip_embeddings.npy         # Saved embeddings
│── main.py                     # Main script
│── README.md                   # This file

---

## ⚙️ Requirements

- Python 3.8+
- Install dependencies:

pip install torch torchvision torchaudio
pip install git+https://github.com/openai/CLIP.git
pip install faiss-cpu pillow tqdm scikit-learn pandas pdfkit
brew install wkhtmltopdf    # (macOS) Required for PDF export

---

## 🚀 Usage

### 1️⃣ Prepare your data

- Place images (`.jpg`, `.jpeg`, `.png`) in `FinalDesignImages/`
- Prepare `design_metadata.xlsx` with these columns:
  - `ImageName` → filename of image (with extension)
  - Other metadata columns (Classification, Make, Collection, Purity, etc.)
  - Exclude unnecessary fields (`Image URL`, `Exists`, `Cpath`) as they won’t display.

---

### 2️⃣ Run script for the first time (build index)

python main.py

---

### 3️⃣ Run in restore mode (skip recomputation)

Edit the script:

RESTORE_MODE = True

Or (if argument support is added):

python main.py --restore

---

### 4️⃣ Export PDF report (optional)

import pdfkit
pdfkit.from_file("clip_ivf_visualizer.html", "clip_ivf_visualizer.pdf")

---

## 📌 Output Example

The HTML viewer shows:

- A **query image** (highlighted in blue)
- Top similar images with:
  - **File name**
  - **Similarity score**
  - **Metadata fields** from Excel

### 📸 Sample Screenshot

![Sample Output Screenshot](docs/sample_viewer_screenshot.png)

*(Replace this placeholder image with your actual HTML screenshot)*

---

## 🔧 Configuration

You can adjust these parameters in `main.py`:

| Parameter              | Description                                              | Default |
|------------------------|----------------------------------------------------------|---------|
| `IMAGE_DIR`           | Folder containing images                                 | FinalDesignImages |
| `EXCEL_PATH`          | Path to Excel file with metadata                         | design_metadata.xlsx |
| `TOP_K`               | Number of top similar matches per image                  | 5       |
| `SIMILARITY_THRESHOLD`| Minimum similarity (0–1 scale, e.g., 0.96 = 96%)         | 0.96    |
| `NLIST`               | Number of FAISS clusters                                 | 50      |
| `RESTORE_MODE`        | Set True to skip recomputation and load saved index       | False   |

---

## 💡 Notes

- CLIP similarity values range **0.0 – 1.0** (not 100).  
- PNG transparency is filled with a white background for consistent embeddings.  
- FAISS IVF requires enough samples; else, it automatically uses FlatIP.  
- Matching ignores **file extensions** (case-insensitive filename match only).  

---

## 📜 License

This project is free to use under the **MIT License**.
