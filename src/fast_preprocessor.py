"""
FAST Preprocessor - Xử lý nhanh, đơn giản nhưng hiệu quả
"""
import cv2
import numpy as np
from PIL import Image


class FastPreprocessor:
    """Preprocessor tối ưu tốc độ"""
    
    def __init__(self):
        pass
    
    def preprocess_for_mnist(self, image):
        """Fast MNIST preprocessing - IMPROVED for digits"""
        if isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image.copy()
        
        # Convert to grayscale
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Invert if needed (MNIST = white digit on black background)
        if np.mean(img) > 127:
            img = 255 - img
        
        # Find bounding box of digit
        _, thresh = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            
            # Add padding
            padding = 20
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(img.shape[1], x + w + padding)
            y2 = min(img.shape[0], y + h + padding)
            
            # Crop
            cropped = img[y1:y2, x1:x2]
            
            if cropped.size > 0:
                img = cropped
        
        # Resize to 28x28 maintaining aspect ratio
        h, w = img.shape
        if h > w:
            new_h = 20
            new_w = max(1, int(w * 20 / h))
        else:
            new_w = 20
            new_h = max(1, int(h * 20 / w))
        
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Center on 28x28 canvas
        canvas = np.zeros((28, 28), dtype=np.uint8)
        x_offset = (28 - new_w) // 2
        y_offset = (28 - new_h) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        # Normalize
        canvas = canvas.astype('float32') / 255.0
        canvas = canvas.reshape(1, 28, 28, 1)
        
        return canvas
    
    def preprocess_for_shape(self, image):
        """Fast Shape preprocessing - 48x48"""
        if isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image.copy()
        
        # Convert to grayscale
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Threshold
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        
        # Resize to 48x48
        img = cv2.resize(img, (48, 48), interpolation=cv2.INTER_AREA)
        
        # Normalize
        img = img.astype('float32') / 255.0
        img = img.reshape(1, 48, 48, 1)
        
        return img
    
    def smart_detect_type(self, image):
        """Smart type detection - IMPROVED"""
        if isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image.copy()
        
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        # Invert if needed
        if np.mean(gray) > 127:
            gray = 255 - gray
        
        # Threshold
        _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 'mnist'
        
        # Largest contour
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        
        if perimeter == 0:
            return 'mnist'
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / h if h > 0 else 1
        
        # Circularity
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Convexity
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Approximate polygon
        epsilon = 0.04 * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        vertices = len(approx)
        
        # Decision logic
        # Perfect circle: high circularity + high solidity + large area
        if circularity > 0.85 and solidity > 0.95 and area > 8000:
            return 'shape'
        
        # Square: 4 vertices + square aspect ratio + high solidity + large area
        if vertices == 4 and 0.8 < aspect_ratio < 1.25 and solidity > 0.9 and area > 7000:
            return 'shape'
        
        # Triangle: 3 vertices + large area
        if vertices == 3 and area > 6000:
            return 'shape'
        
        # Digit: everything else (complex shape, small area, many vertices, low solidity)
        return 'mnist'


_preprocessor = None

def get_fast_preprocessor():
    """Singleton"""
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = FastPreprocessor()
    return _preprocessor