import cv2
import numpy as np
import time
import os
# Initialize the camera
camera = cv2.VideoCapture(0)  # Use the appropriate camera index if necessary
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
    orb = cv2.ORB_create()
    query_keypoints, query_descriptors = orb.detectAndCompute(query_img, None)
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
            if confidence_score > confidence_threshold:
                affected_count += 1
                affected_status = "Affected"
            else:
                affected_status = "Not Affected"
            print(f"Matches with {image_file} - Confidence Score: {confidence_score:.2f} - {affected_status}")
            # Draw the matches for visualization
            result_img = cv2.drawMatches(query_img, query_keypoints, dataset_img, dataset_keypoints, matches[:20], None)
            cv2.imshow(f'Matches with {image_file} - {affected_status} (Confidence: {confidence_score:.2f})', result_img)
            cv2.waitKey(0)
    print(f'Number of affected leaves: {affected_count} out of {len(os.listdir(dataset_folder_path))}')
    cv2.destroyAllWindows()
def capture_and_process_images(dataset_folder_path, capture_interval=120):
    """Capture images from the camera at regular intervals and process them."""
    while True:
        # Capture an image
        ret, frame = camera.read()
        if not ret:
            print("Failed to grab frame.")
            break
        # Save the captured image with a timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        image_filename = f'/home/pi/captured_images/image_{timestamp}.jpg'
        cv2.imwrite(image_filename, frame)
        print(f"Captured and saved image: {image_filename}")
        # Process the captured image
        compare_images(frame, dataset_folder_path)
        # Wait for the specified interval before capturing the next image
        time.sleep(capture_interval)
# Set the dataset folder path
dataset_folder_path = '/home/studen/Downloads/Miner-20210326T082341Z-001/Miner'  # Replace with your dataset folder path
# Create directory for captured images if it doesn't exist
if not os.path.exists('/home/pi/captured_images'):
    os.makedirs('/home/pi/captured_images')
# Start capturing and processing images at regular intervals
capture_and_process_images(dataset_folder_path)