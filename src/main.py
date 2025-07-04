import os
import re
from plate_detection import detect_and_read_plate, display_results

LABELS_FILE = os.path.join('data/labeled_data', 'labels.txt')
image_path = "data/raw_images"

def save_label(image_path, plate_text, labels_file=LABELS_FILE):
    """Save the mapping of image path to license plate text."""
    with open(labels_file, 'a') as f:
        f.write(f"{image_path}\t{plate_text}\n")

def main(img):
    # img_path = os.path.join(image_path, img)
    result = detect_and_read_plate(img)
    display_results(result)
    if result['success']:
        print(f"Detected Text: {result['plate_text']}")
        print(f"Detection Confidence: {result['detection_confidence']:.2f}")
        print(f"Overall Confidence: {result['confidence']:.2f}")

        # Prompt user for correction
        user_input = input(f"Enter correct license plate number for {os.path.basename(img)} [{result['plate_text']}]: ").strip()
        if not user_input:
            user_input = result['plate_text']
        save_label(img, user_input)
    
# Train the model with the labeled data
    # Assuming you have a function to train the model
    # train_model(LABELS_FILE)
if __name__ == "__main__":
    # Regex pattern: ends with .jpg or .jpeg
    pattern = re.compile(r'\.(jpg|jpeg|png)$', re.IGNORECASE)
    for root, dirs, files in os.walk(image_path):
        print("Directory:", dirs)
        for file in files:
            lower_file = file.lower()
            # Check for double image extensions
            extensions = ['.jpg', '.jpeg', '.png']
            ext_count = sum([lower_file.count(ext) for ext in extensions])
            if ext_count > 1:
                # Remove the last extension
                base, ext = os.path.splitext(file)
                new_file = base
                # If still ends with an image extension, keep it
                for ext2 in extensions:
                    if new_file.lower().endswith(ext2):
                        break
                else:
                    # If not, add back the previous extension
                    new_file = base + ext
                old_path = os.path.join(root, file)
                new_path = os.path.join(root, new_file)
                if not os.path.exists(new_path):
                    os.rename(old_path, new_path)
                    file = new_file
                else:
                    file = new_file  # fallback if already exists
            if pattern.search(file):
                print(os.path.join(root, file))
                main(os.path.join(root, file))