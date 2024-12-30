import cv2
import numpy as np
from pathlib import Path
import pickle
from PIL import Image
from torchvision import transforms
import os

class MandelbrotFeatureMatcher:
    def __init__(self, use_sift=True):
        """
        Initialize the feature matcher for Mandelbrot set images.
        
        Why SIFT or SURF?
        - Both are scale and rotation invariant feature detectors
        - SIFT (Scale-Invariant Feature Transform) is generally more accurate but slower
        - SURF (Speeded-Up Robust Features) is faster but might be less accurate
        
        Parameters:
            use_sift (bool): If True, use SIFT detector; if False, use SURF
        """
        # Initialize feature detector (SIFT or SURF)
        if use_sift:
            self.detector = cv2.SIFT_create()
        else:
            self.detector = cv2.SURF_create()
            
        # FLANN matcher parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Database of features and their corresponding coordinates
        self.features_db = []
        self.coordinates_db = []
        
    def add_reference_image(self, image, coordinates):
        """
        Process a reference image and add its features to our database.
        
        Workflow:
        1. Convert image to grayscale (features work better on grayscale)
        2. Detect keypoints (interesting points in the image)
        3. Compute descriptors (numerical description of the area around each keypoint)
        4. Store both descriptors and corresponding coordinates
        
        Parameters:
            image: Input image (can be color or grayscale)
            coordinates: Dict containing x, y coordinates and zoom level
        """
        # Convert image to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Detect keypoints and compute descriptors
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        
        if descriptors is not None and len(keypoints) > 0:
            self.features_db.append({
                'keypoints': keypoints,
                'descriptors': descriptors
            })
            self.coordinates_db.append(coordinates)
            
    def save_database(self, filepath):
        """
        Save the feature database to a file.
        
        Challenge:
        - KeyPoint objects aren't directly serializable (can't be pickled)
        - Need to convert KeyPoints to basic Python types first
        
        Parameters:
            filepath: Where to save the database
        """
        # Convert keypoints to serializable format
        serializable_db = []
        for entry in self.features_db:
            keypoints = [(kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id) 
                        for kp in entry['keypoints']]
            serializable_db.append({
                'keypoints': keypoints,
                'descriptors': entry['descriptors']
            })
            
        # Save database and coordinates
        with open(filepath, 'wb') as f:
            pickle.dump({
                'features': serializable_db,
                'coordinates': self.coordinates_db
            }, f)
            
    def load_database(self, filepath):
        """
        Load a previously saved feature database.
        
        Process:
        1. Load the pickled data
        2. Convert stored keypoint data back into KeyPoint objects
        3. Restore the database structure
        
        Parameters:
            filepath: Path to the saved database
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        self.coordinates_db = data['coordinates']
        self.features_db = []
        
        # Convert back to keypoints
        for entry in data['features']:
            keypoints = [cv2.KeyPoint(x=kp[0][0], y=kp[0][1], 
                                    size=kp[1], angle=kp[2],
                                    response=kp[3], octave=kp[4],
                                    class_id=kp[5]) 
                        for kp in entry['keypoints']]
            self.features_db.append({
                'keypoints': keypoints,
                'descriptors': entry['descriptors']
            })
            
    def find_location(self, query_image, min_matches=5):
        """
        Find the location of a query image in the Mandelbrot set.
        
        Algorithm:
        1. Extract features from query image
        2. Match against all reference images
        3. Use ratio test to filter good matches
        4. Return coordinates of best matching reference image
        
        The ratio test (Lowe's ratio test):
        - For each feature, find the two closest matches
        - If first match is significantly better than second (distance ratio < 0.7)
        - Then it's probably a good match
        
        Parameters:
            query_image: Image to locate
            min_matches: Minimum number of good matches required
            
        Returns:
            coordinates: Dict with x, y, zoom or None if no good match
            match_count: Number of good matches found
        """
        # Prepare query image
        if len(query_image.shape) == 3:
            gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = query_image
            
        # Get features from query image
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        
        if descriptors is None:
            print("No features found in query image")
            return None, 0
            
        print(f"Found {len(keypoints)} keypoints in query image")
        
        best_match = None
        best_match_count = 0
        matches_info = []
        
        # Try to match against each reference image
        for idx, entry in enumerate(self.features_db):
            try:
                # Check feature count for k=2 matching
                if len(entry['descriptors']) < 2:
                    print(f"Warning: Reference image {idx} has too few features")
                    continue
                    
                # Find 2 nearest neighbors for each descriptor
                matches = self.matcher.knnMatch(descriptors, entry['descriptors'], k=2)
                
                # Apply the lowe ratio test to filter good matches
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) < 2:
                        continue
                    m, n = match_pair
                    # If best is much better than second best keep
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
                        
                matches_info.append({
                    'index': idx,
                    'matches': len(good_matches),
                    'good_matches': good_matches
                })
                
                # Keep track of best match
                if len(good_matches) > best_match_count:
                    best_match_count = len(good_matches)
                    best_match = idx
                    
            except cv2.error as e:
                print(f"Warning: Error matching with reference image {idx}: {str(e)}")
                continue
                
        # Return None if we don't have enough good matches
        if best_match_count < min_matches:
            print(f"Not enough good matches found (found {best_match_count}, need {min_matches})")
            return None, 0
            
        return self.coordinates_db[best_match], best_match_count