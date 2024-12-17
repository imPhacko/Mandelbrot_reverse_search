import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from feature_matching.feature_matcher import MandelbrotFeatureMatcher
from feature_matching.utils import draw_matches
import random
import os

class FeatureMatchingEvaluator:
    def __init__(self, data_dir='data/large_dataset', results_dir='feature_matching_evaluation_results1', train_split=0.8):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        
        # Create results directory if it doesn't exist
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize feature matcher
        self.matcher = MandelbrotFeatureMatcher(use_sift=True)
        
        # Load and prepare data
        self.image_paths = sorted(list(self.data_dir.glob('*.png')))
        print(f"Found {len(self.image_paths)} total images")
        
        # Split into training and test sets
        random.seed(42)  # For reproducibility
        random.shuffle(self.image_paths)
        split_idx = int(len(self.image_paths) * train_split)
        self.train_images = self.image_paths[:split_idx]
        self.test_images = self.image_paths[split_idx:]
        
        print(f"Using {len(self.train_images)} images for training database")
        print(f"Testing on {len(self.test_images)} images")
        
        # Create database with training images only
        print("\nCreating reference database from training images...")
        for img_path in self.train_images:
            # Load image in grayscale
            image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Warning: Could not load {img_path}")
                continue
                
            # Get coordinates
            coords = self.load_coordinates(img_path)
            if coords:
                self.matcher.add_reference_image(image, coords)
        
        print(f"Added {len(self.matcher.features_db)} images to reference database")
        
    def load_coordinates(self, image_path):
        """Load coordinates from corresponding txt file"""
        txt_path = image_path.with_suffix('.txt')
        try:
            with open(txt_path, 'r') as f:
                lines = f.readlines()
                center = complex(lines[0].split(': ')[1].strip())
                zoom = float(lines[1].split(': ')[1].strip())
            return {'x': center.real, 'y': center.imag, 'zoom': zoom}
        except Exception as e:
            print(f"Error loading coordinates for {image_path}: {e}")
            return None
        
    def evaluate_all(self):
        """Evaluate on test set only"""
        results = []
        failed_images = []
        
        print("\nEvaluating on test set...")
        for i, img_path in enumerate(self.test_images):
            print(f"\nTesting image {i+1}/{len(self.test_images)}: {img_path.name}")
            result = self.evaluate_single_image(img_path, show_visualization=True)
            if result:
                results.append(result)
            else:
                failed_images.append(img_path.stem)
                
        # Calculate statistics
        if results:
            avg_pos_error = np.mean([r['pos_error'] for r in results])
            avg_zoom_error = np.mean([r['zoom_error'] for r in results])
            avg_matches = np.mean([r['matches'] for r in results])
            
            print("\nTest Set Results:")
            print(f"Total test images: {len(self.test_images)}")
            print(f"Successful matches: {len(results)}")
            print(f"Failed matches: {len(failed_images)}")
            print(f"Success rate: {len(results)/len(self.test_images)*100:.1f}%")
            print(f"Average position error: {avg_pos_error:.6f}")
            print(f"Average zoom error: {avg_zoom_error:.6f}")
            print(f"Average matches found: {avg_matches:.1f}")
            
            if failed_images:
                print("\nFailed images:")
                for img in failed_images:
                    print(f"- {img}")
            
            # Save overall statistics plot
            self.plot_error_distribution(results)
        else:
            print("No successful matches to analyze!")

    def evaluate_single_image(self, image_path, show_visualization=True):
        """Evaluate matching performance on a single test image"""
        # Load and process image
        query_image = cv2.imread(str(image_path))
        if query_image is None:
            print("Failed to load image")
            return None
            
        # Get actual coordinates
        actual_coords = self.load_coordinates(image_path)
        if not actual_coords:
            return None
            
        # Find location
        pred_coords, match_count = self.matcher.find_location(query_image)
        
        if pred_coords is None:
            print("No match found!")
            return None
            
        # Calculate errors
        pos_error = np.sqrt((pred_coords['x'] - actual_coords['x'])**2 + 
                           (pred_coords['y'] - actual_coords['y'])**2)
        zoom_error = abs(pred_coords['zoom'] - actual_coords['zoom'])
        
        print(f"Found {match_count} matches")
        print(f"Position error: {pos_error:.6f}")
        print(f"Zoom error: {zoom_error:.6f}")
        
        if show_visualization:
            self.visualize_results(query_image, pred_coords, actual_coords, 
                                 image_path.stem)
            
        return {
            'image_name': image_path.stem,
            'matches': match_count,
            'pos_error': pos_error,
            'zoom_error': zoom_error
        }
        
    def visualize_results(self, query_image, pred_coords, actual_coords, image_name):
        """Visualize the matching results and save to file"""
        plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(131)
        plt.imshow(cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB))
        plt.title("Query Image")
        
        # Predicted vs Actual coordinates
        plt.subplot(132)
        plt.scatter([pred_coords['x']], [pred_coords['y']], 
                    c='r', label='Predicted', marker='o', s=100)
        plt.scatter([actual_coords['x']], [actual_coords['y']], 
                    c='g', label='Actual', marker='x', s=100)
        plt.legend()
        plt.title("Position Comparison")
        
        # Add error value to the plot
        pos_error = np.sqrt((pred_coords['x'] - actual_coords['x'])**2 + 
                           (pred_coords['y'] - actual_coords['y'])**2)
        plt.text(0.05, 0.95, f'Error: {pos_error:.6f}', 
                 transform=plt.gca().transAxes,
                 verticalalignment='top')
        
        # Zoom comparison
        plt.subplot(133)
        plt.bar(['Predicted', 'Actual'], 
                [pred_coords['zoom'], actual_coords['zoom']])
        plt.title("Zoom Comparison")
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig(self.results_dir / f"{image_name}_results.png", dpi=300)
        plt.close()
        
    def plot_error_distribution(self, results):
        """Plot distribution of errors and save to file"""
        plt.figure(figsize=(15, 5))
        
        # Position error distribution
        plt.subplot(131)
        plt.hist([r['pos_error'] for r in results], bins=20)
        plt.title("Position Error Distribution")
        plt.xlabel("Error")
        plt.ylabel("Count")
        
        # Zoom error distribution
        plt.subplot(132)
        plt.hist([r['zoom_error'] for r in results], bins=20)
        plt.title("Zoom Error Distribution")
        plt.xlabel("Error")
        plt.ylabel("Count")
        
        # Match count distribution
        plt.subplot(133)
        plt.hist([r['matches'] for r in results], bins=20)
        plt.title("Match Count Distribution")
        plt.xlabel("Matches")
        plt.ylabel("Count")
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "overall_statistics.png")
        plt.close()

def main():
    evaluator = FeatureMatchingEvaluator(
        data_dir='data/large_dataset',
        results_dir='feature_matching_evaluation_results1',
        train_split=0.8
    )
    evaluator.evaluate_all()

if __name__ == "__main__":
    main()
