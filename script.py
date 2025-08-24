import cv2
import os
import glob
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Split or combine character images.")
parser.add_argument('--mode', choices=['split', 'combine'], default='split', help='Mode to run: split or combine')
parser.add_argument('--split_input', type=str, default='split', help='Input folder for split mode')
parser.add_argument('--split_output', type=str, default='combined', help='Output folder for split mode')
parser.add_argument('--combine_input', type=str, default='split', help='Input folder for combine mode')
parser.add_argument('--combine_output', type=str, default='combined', help='Output folder for combine mode')
parser.add_argument('--original_input', type=str, default='test', help='Folder containing original images for combine mode')
parser.add_argument('--padding', type=int, default=10, help='Padding to add around each character')
parser.add_argument('--char_size', type=int, default=64, help='Uniform size for character images (width and height)')
parser.add_argument('--debug', action='store_true', help='Save debug images showing detected character boundaries')
args = parser.parse_args()

if args.mode == "split":
    INPUT_FOLDER = args.split_input
    OUTPUT_FOLDER = args.split_output
    PADDING = args.padding
    CHAR_SIZE = args.char_size
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Clear metadata file at start of split mode
    meta_path = os.path.join(OUTPUT_FOLDER, "metadata.txt")
    if os.path.exists(meta_path):
        os.remove(meta_path)  # Remove existing metadata file

    # Loop through all PNG files
    for filepath in glob.glob(os.path.join(INPUT_FOLDER, "*.jpg")):
        filename = os.path.basename(filepath)
        print(f"Processing {filename}...")
        
        # Example filename: 0_unobtainable_0.png
        # Extract label (the word between underscores)
        parts = filename.split("_")
        if len(parts) < 2:
            print(f"Skipping {filename}, filename format not matching")
            continue
        label = parts[1]  # "unobtainable"
        
        # Load image (grayscale)
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Failed to load {filename}")
            continue

        # Find contours on inverted image to get character bounding boxes (original working method)
        thresh = 255 - img  # bitwise inversion
        _, thresh_bin = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sort contours left-to-right
        contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
        
        # Filter contours with original size requirements
        valid_contours = []
        for ctr in contours:
            x, y, w, h = cv2.boundingRect(ctr)
            if w > 2 and h > 2:  # Original filter - this was working
                valid_contours.append(ctr)
        
        print(f" -> Found {len(valid_contours)} valid contours for '{label}' (expected {len(label)})")
        
        # Just warn about mismatches without changing the detection
        if len(valid_contours) != len(label):
            if len(valid_contours) < len(label):
                print(f" -> WARNING: Missing {len(label) - len(valid_contours)} characters in {filename}")
                print(f"    Some characters may not be detected properly")
            else:
                print(f" -> WARNING: Found {len(valid_contours) - len(label)} extra contours")
                print(f"    Extra contours will be ignored")

        # Prepare metadata list - this will store exact crop regions and original image info
        metadata = []
        
        # Store original image dimensions for perfect reconstruction
        h_orig, w_orig = img.shape[:2]
        metadata.append(f"ORIGINAL_SIZE {w_orig} {h_orig}")

        # Save each character (using original working logic)
        for i, ctr in enumerate(valid_contours):
            x, y, w, h = cv2.boundingRect(ctr)
            if i >= len(label):
                print(f"Skipping extra contour {i} in {filename} (no matching label)")
                break
                
            # Extract exact character region from original image
            char_img = img[y:y+h, x:x+w].copy()
            
            # Store original character dimensions before any processing
            orig_char_h, orig_char_w = char_img.shape[:2]
            
            # Add padding around the character
            char_img_padded = cv2.copyMakeBorder(
                char_img, PADDING, PADDING, PADDING, PADDING, 
                cv2.BORDER_CONSTANT, value=255
            )
            
            # Resize to uniform size for model training
            char_img_uniform = cv2.resize(char_img_padded, (CHAR_SIZE, CHAR_SIZE))
            
            char_label = label[i]
            print(f" -> Saving char {i} ({char_label}) from {filename} - original: {orig_char_w}x{orig_char_h}, uniform: {CHAR_SIZE}x{CHAR_SIZE}")

            # Save with new filename: word_charIndex_char.png
            out_path = os.path.join(
                OUTPUT_FOLDER, f"{filename[:-4]}_{i}_{char_label}.png"
            )
            cv2.imwrite(out_path, char_img_uniform)

            # Save metadata: character index, original bounding box (x,y,w,h), padding info, and uniform size
            metadata.append(f"CHAR {i} {x} {y} {w} {h} {PADDING} {CHAR_SIZE}")
        
        # Save debug image only if debug flag is enabled
        if args.debug:
            debug_img = img.copy()
            if len(debug_img.shape) == 2:  # Convert grayscale to color for colored rectangles
                debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2BGR)
            
            for i, ctr in enumerate(valid_contours):
                if i < len(label):
                    x, y, w, h = cv2.boundingRect(ctr)
                    # Draw rectangle around detected character
                    cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 1)
                    # Put character label
                    cv2.putText(debug_img, label[i], (x, y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
            
            # Save debug image
            debug_path = os.path.join(OUTPUT_FOLDER, f"debug_{filename}")
            cv2.imwrite(debug_path, debug_img)

        # Save metadata to a single combined metadata file
        meta_path = os.path.join(OUTPUT_FOLDER, "metadata.txt")
        with open(meta_path, "a") as f:  # Append mode
            # Write filename header
            f.write(f"FILE {filename[:-4]}\n")
            for line in metadata[1:]:  # Skip the first line as it's already written with FILE header
                f.write(line + "\n")
            # Add metadata for original size from the first line
            f.write(metadata[0] + "\n")
            f.write("END_FILE\n")  # Mark end of this file's metadata

    print("✅ Done! All word images split into uniform-sized characters with padding. Metadata saved to metadata.txt")

elif args.mode == "combine":
    INPUT_FOLDER = args.combine_input
    OUTPUT_FOLDER = args.combine_output
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Read the single metadata file
    meta_file = os.path.join(INPUT_FOLDER, "metadata.txt")
    if not os.path.exists(meta_file):
        print("Error: metadata.txt not found in combine input folder")
        exit(1)
    
    print("Reading metadata from metadata.txt...")
    
    # Parse the combined metadata file
    all_files_metadata = {}
    current_file = None
    
    with open(meta_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0] == "FILE":
                current_file = parts[1]
                all_files_metadata[current_file] = {'original_size': None, 'chars': []}
            elif len(parts) >= 3 and parts[0] == "ORIGINAL_SIZE" and current_file:
                all_files_metadata[current_file]['original_size'] = (int(parts[1]), int(parts[2]))
            elif len(parts) == 8 and parts[0] == "CHAR" and current_file:
                idx, x, y, w, h, padding, uniform_size = parts[1:]
                all_files_metadata[current_file]['chars'].append((int(idx), int(x), int(y), int(w), int(h), int(padding), int(uniform_size)))
            elif parts == ["END_FILE"]:
                current_file = None

    print(f"Found metadata for {len(all_files_metadata)} files for combining.")

    for base_name, file_metadata in all_files_metadata.items():
        print(f"Processing {base_name}...")
        
        original_size = file_metadata['original_size']
        char_metadata = file_metadata['chars']

        if original_size is None:
            print(f"Original size not found in metadata for {base_name}, skipping.")
            continue

        # Create blank white image of original size
        w_orig, h_orig = original_size
        combined_img = np.full((h_orig, w_orig), 255, dtype=np.uint8)

        # For each character, load uniform-sized image, restore original size, remove padding, and place back
        for idx, x, y, w, h, padding, uniform_size in char_metadata:
            # Find character image file
            pattern = os.path.join(INPUT_FOLDER, f"{base_name}_{idx}_*.png")
            char_files = glob.glob(pattern)
            if not char_files:
                print(f"Character image for {base_name} index {idx} not found, skipping.")
                continue
            
            char_img_uniform = cv2.imread(char_files[0], cv2.IMREAD_GRAYSCALE)
            if char_img_uniform is None:
                print(f"Failed to read character image {char_files[0]}, skipping.")
                continue

            # First, resize back from uniform size to padded size
            # Calculate what the padded size should be
            padded_size = (w + 2*padding, h + 2*padding)
            char_img_padded = cv2.resize(char_img_uniform, padded_size)
            
            # Then remove padding to get back original character size
            if padding > 0:
                char_img = char_img_padded[padding:-padding, padding:-padding]
            else:
                char_img = char_img_padded

            # Verify the character image has the expected dimensions
            if char_img.shape != (h, w):
                print(f"Warning: Character {idx} size mismatch after processing. Expected ({h},{w}), got {char_img.shape}")
                # Final resize if still not matching (should rarely happen)
                char_img = cv2.resize(char_img, (w, h))

            # Place character image back into combined_img at exact original position
            combined_img[y:y+h, x:x+w] = char_img
            print(f" -> Placed character {idx} at ({x},{y}) size ({w},{h})")

        # Save combined image
        out_path = os.path.join(OUTPUT_FOLDER, f"{base_name}.png")
        cv2.imwrite(out_path, combined_img)
        print(f"✅ Saved combined image: {out_path}")

        # Verify reconstruction by comparing with original if available
        original_img_path = None
        for ext in [".png", ".jpg", ".jpeg"]:
            candidate = os.path.join(args.original_input, base_name + ext)
            if os.path.exists(candidate):
                original_img_path = candidate
                break
        
        if original_img_path:
            original_img = cv2.imread(original_img_path, cv2.IMREAD_GRAYSCALE)
            if original_img is not None:
                # Compare pixel by pixel
                if np.array_equal(combined_img, original_img):
                    print(f"✅ Perfect reconstruction verified for {base_name}")
                else:
                    diff = np.sum(combined_img != original_img)
                    total_pixels = combined_img.size
                    print(f"⚠️  Reconstruction differs by {diff}/{total_pixels} pixels ({100*diff/total_pixels:.2f}%)")

    print("✅ Done! All images combined back to original form.")

# Usage examples:
# Split mode with debug visualization:
# python3 script.py --mode split --split_input images --split_output characters --padding 10 --char_size 64 --debug

# Split mode without debug:
# python3 script.py --mode split --split_input images --split_output characters --padding 10 --char_size 64

# Combine mode:
# python3 script.py --mode combine --combine_input characters --combine_output reconstructed --original_input images