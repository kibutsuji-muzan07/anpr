from ultralytics import YOLO
import cv2
import numpy as np
import easyocr
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

def read_license_plate_from_xml(xml_path):
    """Read license plate number from XML metadata file"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        # Find the object tag and extract the name (license plate number)
        for obj in root.findall('object'):
            name = obj.find('name')
            if name is not None:
                return name.text.strip()
    except Exception as e:
        print(f"Error reading XML file {xml_path}: {str(e)}")
    return None

def get_training_data(base_dir):
    """Get training data by reading all image-XML pairs"""
    training_data = []
    
    def process_directory(directory):
        """Process a directory and its subdirectories for XML-image pairs"""
        if not directory.exists():
            return
        
        # Process all XML files in current directory
        for xml_file in directory.glob('*.xml'):
            # Get corresponding image file
            img_extensions = ['.jpg', '.jpeg', '.png']
            img_file = None
            for ext in img_extensions:
                possible_img = xml_file.with_suffix(ext)
                if possible_img.exists():
                    img_file = possible_img
                    break
            
            if img_file is None:
                continue
                
            # Read license plate number from XML
            plate_number = read_license_plate_from_xml(xml_file)
            if plate_number:
                training_data.append({
                    'image_path': str(img_file),
                    'plate_number': plate_number
                })
        
        # Process subdirectories
        for subdir in directory.iterdir():
            if subdir.is_dir():
                process_directory(subdir)
    
    # List of root directories to search
    search_dirs = [
        Path(base_dir) / 'raw_images',
        Path(base_dir) / 'raw_images' / 'google_images',
        Path(base_dir) / 'raw_images' / 'video_images',
        Path(base_dir) / 'raw_images' / 'State-wise_OLX'
    ]
    
    # Process each root directory
    for search_dir in search_dirs:
        process_directory(search_dir)
    
    return training_data

def preprocess_plate(plate_region):
    """Apply multiple preprocessing techniques and return a list of processed images"""
    processed_images = []
    
    # Original resized image
    min_width = 300
    aspect_ratio = plate_region.shape[1] / plate_region.shape[0]
    new_width = max(min_width, plate_region.shape[1])
    new_height = int(new_width / aspect_ratio)
    resized = cv2.resize(plate_region, (new_width, new_height))
    processed_images.append(resized)

    # Convert to grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    processed_images.append(gray)

    # Apply different preprocessing techniques
    # 1. Adaptive Thresholding
    adaptive_thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    processed_images.append(adaptive_thresh)

    # 2. Otsu's thresholding
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    processed_images.append(otsu)

    # 3. Contrast Enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    processed_images.append(enhanced)

    # 4. Noise reduction
    denoised = cv2.fastNlMeansDenoising(gray)
    processed_images.append(denoised)

    # 5. Edge enhancement
    kernel = np.array([[-1,-1,-1],
                      [-1, 9,-1],
                      [-1,-1,-1]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    processed_images.append(sharpened)

    return processed_images

def is_valid_plate(text, min_length=5, max_length=10):
    """Check if the detected text matches common license plate patterns"""
    # Remove any whitespace and special characters
    text = ''.join(e for e in text if e.isalnum())
    
    # Check length
    if len(text) < min_length or len(text) > max_length:
        return False, 0.0

    # Check if it contains both letters and numbers
    has_letters = bool(re.search(r'[A-Za-z]', text))
    has_numbers = bool(re.search(r'\d', text))
    
    if not (has_letters and has_numbers):
        return False, 0.0

    # Calculate a confidence score based on character composition
    score = 0.0
    total_chars = len(text)
    num_alphanumeric = sum(c.isalnum() for c in text)
    
    # Score based on the ratio of alphanumeric characters
    ratio_score = num_alphanumeric / total_chars
    score += ratio_score * 0.5

    # Score based on having both letters and numbers
    if has_letters and has_numbers:
        score += 0.5

    return True, score

def correct_plate_ocr(text):
    """Correct common OCR mistakes in Indian license plate format."""
    # Example: HR26DQ5551
    # Replace common misreads in the region where letters are expected
    if len(text) >= 6:
        text = list(text)
        # First 2: letters
        for i in range(2):
            if text[i] == '0': text[i] = 'O'
            if text[i] == '1': text[i] = 'I'
            if text[i] == '5': text[i] = 'S'
        # Next 2: digits
        for i in range(2, 4):
            if text[i] == 'O': text[i] = '0'
            if text[i] == 'D': text[i] = '0'
            if text[i] == 'Q': text[i] = '0'
            if text[i] == 'I': text[i] = '1'
            if text[i] == 'S': text[i] = '5'
        # Next 1-2: letters (positions 4,5)
        for i in range(4, min(6, len(text))):
            if text[i] == '0': text[i] = 'O'
            if text[i] == '1': text[i] = 'I'
            if text[i] == '5': text[i] = 'S'
        # Last 4: digits
        for i in range(-4, 0):
            if text[i] == 'O': text[i] = '0'
            if text[i] == 'D': text[i] = '0'
            if text[i] == 'Q': text[i] = '0'
            if text[i] == 'I': text[i] = '1'
            if text[i] == 'S': text[i] = '5'
        text = ''.join(text)
    return text

def detect_and_read_plate(image_path, conf_threshold=0.5):
    try:
        # Check if image path exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file '{image_path}' does not exist.")

        # Initialize EasyOCR reader with optimal parameters
        reader = easyocr.Reader(['en'], gpu=True)  # Use GPU if available
        
        # Load YOLOv8 model
        model = YOLO('models/license_plate_detector.pt')

        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Image file '{image_path}' could not be read. Please check the file format.")

        # Run YOLOv8 inference with confidence threshold
        results = model(img, conf=conf_threshold)
        
        if len(results[0].boxes) == 0:
            raise ValueError("No license plate detected with sufficient confidence")

        # Get the detection with highest confidence
        boxes = results[0].boxes
        confidences = boxes.conf.cpu().numpy()
        best_idx = np.argmax(confidences)
        box = boxes[best_idx]
        
        # Get detection confidence
        detection_conf = confidences[best_idx]
        if detection_conf < conf_threshold:
            raise ValueError(f"Best detection confidence {detection_conf:.2f} below threshold {conf_threshold}")
            
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # Draw rectangle around the license plate
        marked_image = img.copy()
        cv2.rectangle(marked_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        # Add confidence score to the image
        cv2.putText(marked_image, f'Conf: {detection_conf:.2f}', (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        # Crop the license plate region
        plate_region = img[y1:y2, x1:x2]
        
        # Get all preprocessed versions of the plate
        processed_plates = preprocess_plate(plate_region)
        
        # Try OCR on all processed images
        best_text = ""
        best_confidence = 0.0
        best_processed_img = None

        for proc_img in processed_plates:
            # Read with EasyOCR with paragraph detection disabled
            try:
                results = reader.readtext(proc_img, paragraph=False, 
                                        width_ths=1.0,  # Width threshold
                                        height_ths=0.8,  # Height threshold
                                        slope_ths=0.3,   # Allow some rotation
                                        ycenter_ths=0.7, # Vertical alignment threshold
                                        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789') # Only allow alphanumeric
                
                if results:
                    # Process each detected text
                    for (bbox, text, conf) in results:
                        # Clean the text
                        clean_text = ''.join(e for e in text if e.isalnum())
                        # Apply correction
                        corrected_text = correct_plate_ocr(clean_text)
                        # Check if it's a valid plate
                        is_valid, pattern_score = is_valid_plate(corrected_text)
                        if is_valid:
                            # Combine detection confidence, OCR confidence, and pattern matching score
                            total_confidence = (detection_conf + conf + pattern_score) / 3
                            if total_confidence > best_confidence:
                                best_confidence = total_confidence
                                best_text = corrected_text
                                best_processed_img = proc_img
            except Exception as e:
                continue

        if not best_text:
            raise ValueError("Could not read text from license plate")

        # Save the results
        # output_dir = os.path.dirname(image_path)
        # cv2.imwrite(os.path.join(output_dir, 'detected_plate.jpg'), best_processed_img)
        # cv2.imwrite(os.path.join(output_dir, 'marked_car.jpg'), marked_image)

        return {
            'success': True,
            'plate_text': best_text,
            'confidence': best_confidence,
            'detection_confidence': detection_conf,
            'plate_image': best_processed_img,
            'marked_image': marked_image,
            'coordinates': {
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2
            }
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

# def display_results(result):
#     """Display the results of license plate detection"""
#     if not result['success']:
#         print(f"Error: {result['error']}")
#         return

#     # print(f"Detected License Plate Text: {result['plate_text']}")
#     # print(f"Detection Confidence: {result['detection_confidence']:.2f}")
#     # print(f"Overall Confidence: {result['confidence']:.2f}")
    
#     # Display the results
#     cv2.imshow('Original Image with Plate', result['marked_image'])
#     cv2.imshow('License Plate', result['plate_image'])
    
#     # Wait for key press and handle window closing
#     while True:
#         key = cv2.waitKey(1) & 0xFF
#         # If 'q' or ESC is pressed, or window is closed
#         if key == ord('q') or key == 27 or cv2.getWindowProperty('Original Image with Plate', cv2.WND_PROP_VISIBLE) < 1:
#             break
    
#     # Properly destroy all windows
#     cv2.destroyAllWindows()
#     # Ensure windows are actually destroyed by forcing a window update
#     cv2.waitKey(1)

# # Example usage
# if __name__ == "__main__":
#     # Get the training data
#     data_dir = 'data'
#     training_data = get_training_data(data_dir)
#     print(f"Found {len(training_data)} image-label pairs")
    
#     # Process a few examples to verify
#     for item in training_data[:5]:  # Show first 5 examples
#         print(f"\nProcessing {item['image_path']}")
#         print(f"XML License Plate: {item['plate_number']}")
        
#         # Try to detect and read the plate
#         result = detect_and_read_plate(item['image_path'])
#         if result['success']:
#             print(f"Detected Text: {result['plate_text']}")
#             print(f"Detection Confidence: {result['detection_confidence']:.2f}")
#             print(f"Overall Confidence: {result['confidence']:.2f}")
#             display_results(result)