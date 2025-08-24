import os
import shutil

# Folder containing your images
folder = "out"
output_folder = "renamed_out"
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(folder):
    print(f"Processing file: {filename}")
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        # Example: point_1572.png
        name, ext = os.path.splitext(filename)
        
        # Extract the number part (after underscore)
        if "_" in name:
            parts = name.split("_")
            print(f"Parts: {parts}")
            if len(parts) == 2 and parts[1].isdigit():
                number = parts[1]
                print(f"Number extracted: {number}")
                new_name = f"{number}_{name}{ext}"  # 1572_point_1572.png
                old_path = os.path.join(folder, filename)
                new_path = os.path.join(output_folder, new_name)
                print(f"Old path: {old_path}, New path: {new_path}")
                shutil.copy(old_path, new_path)
                print(f"Saved renamed file in renamed_out: {new_name}")
                out_new_path = os.path.join(folder, new_name)
                shutil.copy(new_path, out_new_path)
                print(f"Copied to out: {out_new_path}")