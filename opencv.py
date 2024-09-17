import cv2
import os
import numpy as np
def extract_features(image_path):
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        return None, None
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(img, None)
    return keypoints, descriptors

def compare_images(query_image_path, dataset_folder_path):
    
    query_img = cv2.imread(query_image_path)
    if query_img is None:
        print(f"Could not load query image: {query_image_path}")
        return
    
    orb = cv2.ORB_create()
    query_keypoints, query_descriptors = orb.detectAndCompute(query_img, None)
    print(f"Query image keypoints: {len(query_keypoints)}")
    
    affected_count = 0
    confidence_threshold = 0.5  
    
    for image_file in os.listdir(dataset_folder_path):
        dataset_image_path = os.path.join(dataset_folder_path, image_file)
        
        dataset_keypoints, dataset_descriptors = extract_features(dataset_image_path)
        if dataset_descriptors is None:
            continue  
        print(f"Dataset image keypoints: {len(dataset_keypoints)}")
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
        matches = bf.match(query_descriptors, dataset_descriptors)
        matches = sorted(matches, key=lambda x: x.distance)
        
        if matches:
            average_distance = np.mean([m.distance for m in matches])
            
            max_distance = 100  
            confidence_score = 1 - (average_distance / max_distance) 
            confidence_score = max(0, min(confidence_score, 1))  
            
            if confidence_score > confidence_threshold:
                affected_count += 1
                affected_status = "Affected"
            else:
                affected_status = "Not Affected"
            print(f"Matches with {image_file} - Confidence Score: {confidence_score:.2f} - {affected_status}")
            
            result_img = cv2.drawMatches(query_img, query_keypoints, cv2.imread(dataset_image_path), dataset_keypoints, matches[:20], None)
            cv2.imshow(f'Matches with {image_file} - {affected_status} (Confidence: {confidence_score:.2f})', result_img)
            cv2.waitKey(0)
    print(f'Number of affected leaves: {affected_count} out of {len(os.listdir(dataset_folder_path))}')
    cv2.destroyAllWindows()

query_image_path = 'image/leaf3.jpeg'  
dataset_folder_path = '/home/studen/Downloads/Miner-20210326T082341Z-001/Miner'  

compare_images(query_image_path, dataset_folder_path)
from django.test import TestCase
from django.core.exceptions import ValidationError
from .models import Pest

class PestModelTest(TestCase):
    def setUp(self):
        """Set up a sample Pest instance for testing."""
        self.pest = Pest.objects.create(
            name="Aphid",
            pest_description="Small, green insect that damages plants."
        )

    def test_pest_creation(self):
        """Test that the Pest object is created correctly."""
        pest = Pest.objects.get(id=self.pest.id)  # Using id instead of pest_id
        self.assertEqual(pest.name, "Aphid")
        self.assertEqual(pest.pest_description, "Small, green insect that damages plants.")

    def test_pest_str_method(self):
        """Test the __str__ method of the Pest model."""
        self.assertEqual(str(self.pest), "Aphid")

    def test_pest_update(self):
        """Test updating a Pest object."""
        self.pest.name = "Whitefly"
        self.pest.pest_description = "Tiny, white insect that infests plants."
        self.pest.save()
        updated_pest = Pest.objects.get(id=self.pest.id)  # Using id instead of pest_id
        self.assertEqual(updated_pest.name, "Whitefly")
        self.assertEqual(updated_pest.pest_description, "Tiny, white insect that infests plants.")

    def test_pest_deletion(self):
        """Test deleting a Pest object."""
        self.pest.delete()
        with self.assertRaises(Pest.DoesNotExist):
            Pest.objects.get(id=self.pest.id)  # Using id instead of pest_id

    def test_name_max_length(self):
        """Test that the name field cannot exceed 255 characters."""
        long_name = "A" * 256
        long_name_pest = Pest(
            name=long_name,
            pest_description="Some description."
        )
        with self.assertRaises(ValidationError):
            long_name_pest.full_clean()

    def test_pest_description_max_length(self):
        """Test that the pest_description field cannot exceed 255 characters."""
        long_description = "A" * 256
        long_description_pest = Pest(
            name="Short name",
            pest_description=long_description
        )
        with self.assertRaises(ValidationError):
            long_description_pest.full_clean()

    def test_missing_name(self):
        """Test that the name field cannot be empty."""
        missing_name_pest = Pest(
            name="",
            pest_description="Some description."
        )
        with self.assertRaises(ValidationError):
            missing_name_pest.full_clean()

    def test_missing_pest_description(self):
        """Test that the pest_description field cannot be empty."""
        missing_description_pest = Pest(
            name="Aphid",
            pest_description=""
        )
        with self.assertRaises(ValidationError):
            missing_description_pest.full_clean()