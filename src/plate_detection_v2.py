from ultralytics import YOLO
import cv2
import numpy as np
import easyocr
import os
import re

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

def detect_and_read_plate_v2(image_path):
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

        # Run YOLOv8 inference
        results = model(img)
        
        if len(results[0].boxes) == 0:
            raise ValueError("No license plate detected in the image")

        # Get the detection with highest confidence
        boxes = results[0].boxes
        confidences = boxes.conf.cpu().numpy()
        best_idx = np.argmax(confidences)
        box = boxes[best_idx]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # Draw rectangle around the license plate
        marked_image = img.copy()
        cv2.rectangle(marked_image, (x1, y1), (x2, y2), (0, 255, 0), 3)

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
                                        width_ths=1.0,  # Adjusted width threshold
                                        height_ths=0.8,  # Adjusted height threshold
                                        slope_ths=0.3,   # Allow some rotation
                                        ycenter_ths=0.7) # Vertical alignment threshold
                
                if results:
                    # Process each detected text
                    for (bbox, text, conf) in results:
                        # Clean the text
                        clean_text = ''.join(e for e in text if e.isalnum())
                        
                        # Check if it's a valid plate
                        is_valid, pattern_score = is_valid_plate(clean_text)
                        
                        if is_valid:
                            # Combine OCR confidence with pattern matching score
                            total_confidence = (conf + pattern_score) / 2
                            
                            if total_confidence > best_confidence:
                                best_confidence = total_confidence
                                best_text = clean_text
                                best_processed_img = proc_img

            except Exception as e:
                continue

        if not best_text:
            raise ValueError("Could not read text from license plate")

        # Save the results
        output_dir = os.path.dirname(image_path)
        cv2.imwrite(os.path.join(output_dir, 'detected_plate.jpg'), best_processed_img)
        cv2.imwrite(os.path.join(output_dir, 'marked_car.jpg'), marked_image)

        return {
            'success': True,
            'plate_text': best_text,
            'confidence': best_confidence,
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
