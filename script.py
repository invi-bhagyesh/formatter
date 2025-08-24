import cv2
import os
import argparse
import numpy as np

def split_images(input_folder, output_folder, padding=5, char_size=64, debug=False):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if not (filename.endswith(".png") or filename.endswith(".jpg")):
            continue

        filepath = os.path.join(input_folder, filename)
        image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

        # Thresholding for segmentation
        _, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)

        # Find contours (characters)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])  # left to right

        char_folder = os.path.join(output_folder, os.path.splitext(filename)[0])
        os.makedirs(char_folder, exist_ok=True)

        for idx, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)

            # Extract character with padding
            char_img = image[max(0, y-padding): y+h+padding, max(0, x-padding): x+w+padding]

            # Resize to fixed size
            char_img = cv2.resize(char_img, (char_size, char_size))

            char_path = os.path.join(char_folder, f"{idx:03d}.png")
            cv2.imwrite(char_path, char_img)

            if debug:
                debug_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.imshow("Debug Split", debug_img)
                cv2.waitKey(200)

    if debug:
        cv2.destroyAllWindows()


def combine_images(input_folder, output_folder, original_input=None):
    os.makedirs(output_folder, exist_ok=True)

    for word_folder in os.listdir(input_folder):
        char_folder = os.path.join(input_folder, word_folder)
        if not os.path.isdir(char_folder):
            continue

        char_files = sorted(os.listdir(char_folder))
        chars = [cv2.imread(os.path.join(char_folder, f), cv2.IMREAD_GRAYSCALE) for f in char_files]

        if not chars:
            continue

        # Concatenate characters horizontally
        combined = np.concatenate(chars, axis=1)

        # If original exists, resize to match its height
        if original_input:
            original_path = os.path.join(original_input, f"{word_folder}.png")
            if os.path.exists(original_path):
                orig_img = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
                combined = cv2.resize(combined, (orig_img.shape[1], orig_img.shape[0]))

        out_path = os.path.join(output_folder, f"{word_folder}.png")
        cv2.imwrite(out_path, combined)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["split", "combine"], required=True)

    parser.add_argument("--split_input", help="Input folder with word images")
    parser.add_argument("--split_output", help="Output folder for characters")
    parser.add_argument("--padding", type=int, default=5, help="Padding around characters")
    parser.add_argument("--char_size", type=int, default=64, help="Resize character size")
    parser.add_argument("--debug", action="store_true", help="Enable debug visualization")

    parser.add_argument("--combine_input", help="Input folder with character subfolders")
    parser.add_argument("--combine_output", help="Output folder for reconstructed words")
    parser.add_argument("--original_input", help="Original word images for size reference")

    args = parser.parse_args()

    if args.mode == "split":
        if not args.split_input or not args.split_output:
            raise ValueError("split mode requires --split_input and --split_output")
        split_images(args.split_input, args.split_output, args.padding, args.char_size, args.debug)

    elif args.mode == "combine":
        if not args.combine_input or not args.combine_output:
            raise ValueError("combine mode requires --combine_input and --combine_output")
        combine_images(args.combine_input, args.combine_output, args.original_input)

# python3 script.py --mode split --split_input images --split_output characters --padding 10 --char_size 64
# python3 script.py --mode combine --combine_input characters --combine_output reconstructed --original_input images
