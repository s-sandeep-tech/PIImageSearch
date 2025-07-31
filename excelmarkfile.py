import os
import pandas as pd

# === CONFIGURATION ===
excel_path = "/Users/sandeeps/Documents/PyImageSearch/designList.xlsx"
image_folder = "/Users/sandeeps/Documents/PyImageSearch/FinalDesignImages"
output_excel = "/Users/sandeeps/Documents/PyImageSearch/designList2.xlsx"

# === LOAD EXCEL ===
df = pd.read_excel(excel_path)

# === CHECK AND ADD COLUMNS ===
def build_cpath(image_name):
    if pd.isna(image_name):  # Skip NaN or empty
        return ""
    base_name = os.path.splitext(str(image_name))[0] + ".jpg"
    return os.path.join(image_folder, base_name)

def check_exists(cpath):
    return "✔️" if cpath and os.path.exists(cpath) else "❌"

# Add Cpath and Exists columns safely
df["Cpath"] = df["ImageName"].apply(build_cpath)
df["Exists"] = df["Cpath"].apply(check_exists)

# === SAVE TO EXCEL ===
df.to_excel(output_excel, index=False)
print(f"✅ Updated Excel saved at: {output_excel}")
