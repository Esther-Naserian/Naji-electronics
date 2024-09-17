import cv2
import numpy as np
import time
import os

# Initialize the camera
camera = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not camera.isOpened():
    print("Error: Could not open camera.")
    exit()

def extract_features(img):
    """Extract features from a given image."""
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(img, None)
    return keypoints, descriptors

def compare_images(query_img, dataset_folder_path):
    """Compare query image against a dataset of images."""
    if query_img is None:
        print("Query image is empty.")
        return
    
    # Extract features from the query image
    query_keypoints, query_descriptors = extract_features(query_img)
    print(f"Query image keypoints: {len(query_keypoints)}")
    
    # Initialize variables for confidence scores
    affected_count = 0
    confidence_threshold = 0.5  # Set a threshold for confidence score
    
    # Loop through each image in the dataset folder
    for image_file in os.listdir(dataset_folder_path):
        dataset_image_path = os.path.join(dataset_folder_path, image_file)
        
        # Load the dataset image
        dataset_img = cv2.imread(dataset_image_path)
        if dataset_img is None:
            continue
        
        # Extract features from the dataset image
        dataset_keypoints, dataset_descriptors = extract_features(dataset_img)
        if dataset_descriptors is None:
            continue
        
        print(f"Dataset image keypoints: {len(dataset_keypoints)}")
        
        # Create a Brute-Force Matcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Match descriptors
        matches = bf.match(query_descriptors, dataset_descriptors)
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Calculate the average distance of the matches
        if matches:
            average_distance = np.mean([m.distance for m in matches])
            max_distance = 100  # Adjust this based on your data
            confidence_score = 1 - (average_distance / max_distance)  # Normalize to [0, 1]
            confidence_score = max(0, min(confidence_score, 1))  # Clamp to [0, 1]
            
            # Determine if the leaf is affected based on confidence score
            affected_status = "Affected" if confidence_score > confidence_threshold else "Not Affected"
            
            print(f"Matches with {image_file} - Confidence Score: {confidence_score:.2f} - {affected_status}")
            
            # Draw the matches for visualization
            result_img = cv2.drawMatches(query_img, query_keypoints, dataset_img, dataset_keypoints, matches[:20], None)
            cv2.imshow(f'Matches with {image_file} - {affected_status} (Confidence: {confidence_score:.2f})', result_img)
            cv2.waitKey(0)

    cv2.destroyAllWindows()

def capture_and_process_image(dataset_folder_path):
    """Capture an image from the camera and process it immediately."""
    # Capture an image
    ret, frame = camera.read()
    if not ret:
        print("Failed to grab frame.")
        return
    
    # Save the captured image with a timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    captured_images_folder = './captured_images'
    
    # Create directory for captured images if it doesn't exist
    if not os.path.exists(captured_images_folder):
        os.makedirs(captured_images_folder)
    
    image_filename = os.path.join(captured_images_folder, f'image_{timestamp}.jpg')
    cv2.imwrite(image_filename, frame)
    print(f"Captured and saved image: {image_filename}")
    
    # Process the captured image
    compare_images(frame, dataset_folder_path)

# Get the current working directory
current_dir = os.getcwd()

# Set the dataset folder path (relative to the current working directory)
dataset_folder_path = os.path.join(current_dir, 'Miner')

# Capture and process a single image in real time
capture_and_process_image(dataset_folder_path)

# Release the camera when done
camera.release()
cv2.destroyAllWindows()
