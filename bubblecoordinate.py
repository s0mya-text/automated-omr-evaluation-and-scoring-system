import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List

class BubbleCoordinateMapper:
    """Generate coordinate map for all bubbles using reference points and spacing"""
    
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # These will be filled with actual measurements
        self.reference_coords = {}
        self.spacing = {}
        self.bubble_map = {}
        
    def set_reference_points(self, q1_a: Tuple[int, int], q1_b: Tuple[int, int], 
                           q2_a: Tuple[int, int], q21_a: Tuple[int, int]):
        """
        Set the reference coordinates found manually
        
        Args:
            q1_a: (x, y) coordinates for Question 1, Choice A
            q1_b: (x, y) coordinates for Question 1, Choice B  
            q2_a: (x, y) coordinates for Question 2, Choice A
            q21_a: (x, y) coordinates for Question 21, Choice A
        """
        self.reference_coords = {
            'q1_a': q1_a,
            'q1_b': q1_b, 
            'q2_a': q2_a,
            'q21_a': q21_a
        }
        
        # Calculate spacing from reference points
        self.spacing = {
            'x_spacing': q1_b[0] - q1_a[0],  # Horizontal distance between choices
            'y_spacing': q2_a[1] - q1_a[1],  # Vertical distance between questions
            'col_spacing': q21_a[0] - q1_a[0]  # Distance to next column (Q21 is start of column 2)
        }
        
        print("Reference points set:")
        print(f"Q1-A: {q1_a}, Q1-B: {q1_b}, Q2-A: {q2_a}, Q21-A: {q21_a}")
        print(f"Calculated spacing - X: {self.spacing['x_spacing']}, Y: {self.spacing['y_spacing']}, Column: {self.spacing['col_spacing']}")
    
    def generate_bubble_map(self, questions_per_column: int = 20, 
                          total_questions: int = 100, choices_per_question: int = 4) -> Dict:
        """
        Generate coordinates for all bubbles based on reference points
        
        Args:
            questions_per_column: Number of questions in each column (default: 20)
            total_questions: Total number of questions (default: 100)
            choices_per_question: Number of choices per question (default: 4)
        """
        if not self.reference_coords:
            raise ValueError("Reference points must be set first using set_reference_points()")
        
        bubble_map = {}
        choices = ['A', 'B', 'C', 'D']
        
        # Calculate number of columns
        num_columns = (total_questions + questions_per_column - 1) // questions_per_column
        
        for question_num in range(1, total_questions + 1):
            # Determine which column this question is in
            column = (question_num - 1) // questions_per_column
            row_in_column = (question_num - 1) % questions_per_column
            
            # Base coordinates for this question (Choice A)
            base_x = self.reference_coords['q1_a'][0] + (column * self.spacing['col_spacing'])
            base_y = self.reference_coords['q1_a'][1] + (row_in_column * self.spacing['y_spacing'])
            
            # Generate coordinates for all choices of this question
            question_bubbles = {}
            for choice_idx, choice in enumerate(choices[:choices_per_question]):
                x = base_x + (choice_idx * self.spacing['x_spacing'])
                y = base_y
                question_bubbles[choice] = {
                    'center': (x, y),
                    'bbox': self._get_bubble_bbox(x, y)
                }
            
            bubble_map[f'Q{question_num}'] = question_bubbles
        
        self.bubble_map = bubble_map
        return bubble_map
    
    def _get_bubble_bbox(self, center_x: int, center_y: int, 
                        bubble_radius: int = 12) -> Tuple[int, int, int, int]:
        """
        Get bounding box coordinates for a bubble given its center
        
        Args:
            center_x, center_y: Center coordinates of the bubble
            bubble_radius: Radius of the bubble in pixels
            
        Returns:
            Tuple of (x, y, width, height) for bounding box
        """
        x = center_x - bubble_radius
        y = center_y - bubble_radius
        width = height = 2 * bubble_radius
        return (x, y, width, height)
    
    def visualize_bubble_map(self, show_question_numbers: bool = True, 
                           sample_questions: List[int] = None):
        """
        Visualize the generated bubble map on the image
        
        Args:
            show_question_numbers: Whether to show question numbers
            sample_questions: List of specific questions to highlight (None = show all)
        """
        if not self.bubble_map:
            raise ValueError("Bubble map not generated. Call generate_bubble_map() first.")
        
        # Create a copy of the image for visualization
        vis_image = self.image.copy()
        
        # Define colors
        colors = {'A': (0, 255, 0), 'B': (255, 0, 0), 'C': (0, 0, 255), 'D': (255, 255, 0)}
        
        questions_to_show = sample_questions if sample_questions else range(1, len(self.bubble_map) + 1)
        
        for question_num in questions_to_show:
            question_key = f'Q{question_num}'
            if question_key not in self.bubble_map:
                continue
                
            for choice, bubble_data in self.bubble_map[question_key].items():
                center = bubble_data['center']
                color = colors[choice]
                
                # Draw circle at bubble center
                cv2.circle(vis_image, center, 8, color, 2)
                
                # Draw bounding box
                x, y, w, h = bubble_data['bbox']
                cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 1)
                
                # Add choice label
                cv2.putText(vis_image, choice, (center[0] - 5, center[1] + 3),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
            # Add question number
            if show_question_numbers:
                first_choice_center = self.bubble_map[question_key]['A']['center']
                cv2.putText(vis_image, str(question_num), 
                           (first_choice_center[0] - 20, first_choice_center[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Display the image
        plt.figure(figsize=(15, 20))
        plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        plt.title("Bubble Coordinate Map Visualization")
        plt.axis('off')
        plt.show()
        
        return vis_image
    
    def save_bubble_map(self, output_path: str = 'bubble_coordinate_map.json'):
        """Save the bubble map to a JSON file"""
        if not self.bubble_map:
            raise ValueError("Bubble map not generated. Call generate_bubble_map() first.")
        
        # Convert to JSON-serializable format
        json_map = {}
        for question, choices in self.bubble_map.items():
            json_map[question] = {}
            for choice, bubble_data in choices.items():
                json_map[question][choice] = {
                    'center_x': int(bubble_data['center'][0]),
                    'center_y': int(bubble_data['center'][1]),
                    'bbox_x': int(bubble_data['bbox'][0]),
                    'bbox_y': int(bubble_data['bbox'][1]),
                    'bbox_width': int(bubble_data['bbox'][2]),
                    'bbox_height': int(bubble_data['bbox'][3])
                }
        
        # Also save the reference points and spacing for future use
        metadata = {
            'reference_coords': self.reference_coords,
            'spacing': self.spacing,
            'image_path': self.image_path
        }
        
        output_data = {
            'metadata': metadata,
            'bubble_map': json_map
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Bubble map saved to {output_path}")
    
    def load_bubble_map(self, json_path: str):
        """Load bubble map from JSON file"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        self.reference_coords = data['metadata']['reference_coords']
        self.spacing = data['metadata']['spacing']
        
        # Convert back to the internal format
        self.bubble_map = {}
        for question, choices in data['bubble_map'].items():
            self.bubble_map[question] = {}
            for choice, bubble_data in choices.items():
                self.bubble_map[question][choice] = {
                    'center': (bubble_data['center_x'], bubble_data['center_y']),
                    'bbox': (bubble_data['bbox_x'], bubble_data['bbox_y'], 
                            bubble_data['bbox_width'], bubble_data['bbox_height'])
                }
        
        print(f"Bubble map loaded from {json_path}")

# Example usage
if __name__ == "__main__":
    # Initialize the mapper with your rectified image
    mapper = BubbleCoordinateMapper('rectified_image.png')
    
    # TODO: Replace these with your actual measured coordinates
    # You need to find these coordinates manually using an image editor
    q1_a_coords = (100, 150)    # Replace with actual Q1-A coordinates
    q1_b_coords = (150, 150)    # Replace with actual Q1-B coordinates  
    q2_a_coords = (100, 180)    # Replace with actual Q2-A coordinates
    q21_a_coords = (300, 150)   # Replace with actual Q21-A coordinates
    
    # Set the reference points
    mapper.set_reference_points(q1_a_coords, q1_b_coords, q2_a_coords, q21_a_coords)
    
    # Generate the complete bubble map
    bubble_map = mapper.generate_bubble_map()
    
    # Visualize a sample of questions to verify accuracy
    mapper.visualize_bubble_map(sample_questions=[1, 2, 3, 21, 22, 23, 41, 42, 43])
    
    # Save the bubble map for future use
    mapper.save_bubble_map('bubble_coordinate_map.json')
    
    # Print some example coordinates
    print("\nExample bubble coordinates:")
    for q in ['Q1', 'Q21', 'Q41', 'Q100']:
        if q in bubble_map:
            print(f"{q}: {bubble_map[q]}")