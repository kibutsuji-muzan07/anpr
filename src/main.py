import os
from pathlib import Path
import xml.etree.ElementTree as ET
from plate_detection import detect_and_read_plate, get_training_data

LABELS_FILE = os.path.join('data/labeled_data', 'labels.txt')
DATA_DIR = "data"

def save_to_labels_file(training_data, labels_file=LABELS_FILE):
    """Save all image paths and license plate numbers to the labels file"""
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(labels_file), exist_ok=True)
    
    # Remove existing labels file if it exists
    if os.path.exists(labels_file):
        os.remove(labels_file)
    
    # Write the data
    with open(labels_file, 'w') as f:
        for item in training_data:
            f.write(f"{item['image_path']}\t{item['plate_number']}\n")

def verify_detections(training_data, num_samples=None):
    """Verify plate detections against XML data"""
    import random
    
    if num_samples is not None:
        # Randomly sample the data
        if num_samples > len(training_data):
            num_samples = len(training_data)
        data_to_process = random.sample(training_data, num_samples)
    else:
        data_to_process = training_data

    results = {
        'total': len(data_to_process),
        'successful_detections': 0,
        'matches': 0,
        'mismatches': 0,
        'failed_detections': 0
    }

    for item in data_to_process:
        print(f"\nProcessing: {item['image_path']}")
        print(f"Ground Truth (XML): {item['plate_number']}")
        
        # Try to detect the plate
        result = detect_and_read_plate(item['image_path'])
        
        if result['success']:
            results['successful_detections'] += 1
            detected_text = result['plate_text']
            print(f"Detected Text: {detected_text}")
            print(f"Confidence: {result['confidence']:.2f}")
            
            # Compare with ground truth
            if detected_text.upper() == item['plate_number'].upper():
                results['matches'] += 1
                print("✓ Match!")
            else:
                results['mismatches'] += 1
                print("✗ Mismatch")
        else:
            results['failed_detections'] += 1
            print(f"Detection failed: {result['error']}")

    return results

def print_statistics(stats):
    """Print detection and matching statistics"""
    print("\n=== Detection Statistics ===")
    print(f"Total images processed: {stats['total']}")
    print(f"Successful detections: {stats['successful_detections']} ({(stats['successful_detections']/stats['total']*100):.1f}%)")
    print(f"Failed detections: {stats['failed_detections']} ({(stats['failed_detections']/stats['total']*100):.1f}%)")
    if stats['successful_detections'] > 0:
        print(f"\nAmong successful detections:")
        print(f"Correct matches: {stats['matches']} ({(stats['matches']/stats['successful_detections']*100):.1f}%)")
        print(f"Mismatches: {stats['mismatches']} ({(stats['mismatches']/stats['successful_detections']*100):.1f}%)")

if __name__ == "__main__":
    # Get all training data from XML files
    print("Reading training data from XML files...")
    training_data = get_training_data(DATA_DIR)
    print(f"Found {len(training_data)} image-label pairs")
    
    # Save all data to labels file
    save_to_labels_file(training_data)
    print(f"Saved all labels to {LABELS_FILE}")
    
    # Verify a sample of detections against XML data
    print("\nVerifying plate detections...")
    sample_size = 10  # Adjust this number to process more or fewer samples
    stats = verify_detections(training_data, sample_size)
    print_statistics(stats)