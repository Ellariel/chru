import os
from striprtf.striprtf import rtf_to_text

input_folder = "./ru_all_rtf"
output_folder = "./ru_all_txt"

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(".rtf"):
        rtf_path = os.path.join(input_folder, filename)
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        txt_path = os.path.join(output_folder, txt_filename)

        with open(rtf_path, "r", encoding="latin-1", errors="ignore") as f:
            rtf_content = f.read()

        text = rtf_to_text(rtf_content)

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"Converted: {filename} → {txt_filename}")