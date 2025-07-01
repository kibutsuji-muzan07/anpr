import os
import re
from plate_detection import detect_and_read_plate, display_results

LABELS_FILE = os.path.join('data/labeled_data', 'labels.txt')
image_path = "data/raw_images"

def save_label(image_path, plate_text, labels_file=LABELS_FILE):
    """Save the mapping of image filename to license plate text."""
    filename = os.path.basename(image_path)
    with open(labels_file, 'a') as f:
        f.write(f"{filename}\t{plate_text}\n")

def main(img):
    img_path = os.path.join(image_path, img)
    result = detect_and_read_plate(img_path)
    display_results(result)
    if result['success']:
        print(f"Detected Text: {result['plate_text']}")
        print(f"Detection Confidence: {result['detection_confidence']:.2f}")
        print(f"Overall Confidence: {result['confidence']:.2f}")

        # Prompt user for correction
        user_input = input(f"Enter correct license plate number for {os.path.basename(img_path)} [{result['plate_text']}]: ").strip()
        if not user_input:
            user_input = result['plate_text']
        save_label(img_path, user_input)
    

if __name__ == "__main__":
    # Regex pattern: ends with _<digit>.jpg or .jpeg
    pattern = re.compile(r'_+\d+\.(jpg|jpeg)$', re.IGNORECASE)
    for img in os.listdir(image_path):  # Adjust range as needed
        if pattern.search(img):
            main(img)