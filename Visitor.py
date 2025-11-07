import numpy as np
import random
import cv2

class Visitor:
    def __init__(self, id, frame_shape):
        self.id = id
        self.path = []  # Track movement path
        self.age = None
        self.gender = None
        self.analyzed = False  # Only analyze once
        self.path_points = []
        self.path_image = np.zeros(frame_shape, dtype=np.uint8) # Blank image
        self.color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))

    def add_point(self, x, y):
        # Add new point
        new_point = (x, y)
        
        # If we have a previous point, draw line to it
        if len(self.path_points) > 0:
            last_point = self.path_points[-1]
            cv2.line(self.path_image, last_point, new_point, self.color, 2)
        
        # Draw circle at new point
        cv2.circle(self.path_image, new_point, 4, self.color, -1)
        
        self.path_points.append(new_point)
    
    def get_path_image(self):
        """Return the accumulated path image"""
        return self.path_image.copy()
