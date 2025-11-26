import cv2
import numpy as np
from PIL import Image

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Reads an image, resizes it, and normalizes it for the model.
    """
    try:
        # Read image using OpenCV
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize
        img = cv2.resize(img, target_size)
        
        # Normalize to [0, 1]
        img = img.astype('float32') / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None
