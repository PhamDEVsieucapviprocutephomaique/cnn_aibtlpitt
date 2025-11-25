"""
Advanced Image Preprocessor - Xử lý ảnh chuyên sâu với các kỹ thuật tiên tiến
"""
import cv2
import numpy as np
from PIL import Image
import hashlib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import *


class AdvancedImageProcessor:
    """Xử lý ảnh với các kỹ thuật computer vision tiên tiến"""
    
    def __init__(self):
        self.cache = {}
    
    def apply_clahe(self, img):
        """Contrast Limited Adaptive Histogram Equalization"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img)
    
    def bilateral_filter(self, img):
        """Bilateral filter - giữ edges, giảm noise"""
        return cv2.bilateralFilter(img, 9, 75, 75)
    
    def morphological_operations(self, img):
        """Morphological operations để làm sạch ảnh"""
        # Closing - đóng các lỗ nhỏ
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        
        # Opening - loại bỏ noise nhỏ
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
        
        return opened
    
    def edge_detection_multi(self, img):
        """Multi-scale edge detection"""
        # Canny edge
        edges_canny = cv2.Canny(img, 50, 150)
        
        # Sobel X và Y
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobelx**2 + sobely**2).astype(np.uint8)
        
        # Laplacian
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        laplacian = np.absolute(laplacian).astype(np.uint8)
        
        # Combine edges
        edges = cv2.bitwise_or(edges_canny, sobel)
        edges = cv2.bitwise_or(edges, laplacian)
        
        return edges
    
    def find_and_center_object(self, img):
        """Tìm và căn giữa object"""
        # Threshold
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return img
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Crop object với padding
        padding = 10
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img.shape[1], x + w + padding)
        y2 = min(img.shape[0], y + h + padding)
        
        cropped = img[y1:y2, x1:x2]
        
        if cropped.size == 0:
            return img
        
        return cropped
    
    def normalize_size(self, img, target_size):
        """Normalize size giữ aspect ratio"""
        h, w = img.shape[:2]
        
        # Tính scale để fit vào target_size
        scale = min(target_size[0] / w, target_size[1] / h)
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Tạo canvas và center object
        canvas = np.zeros(target_size, dtype=img.dtype)
        
        x_offset = (target_size[1] - new_w) // 2
        y_offset = (target_size[0] - new_h) // 2
        
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas
    
    def advanced_threshold(self, img):
        """Advanced thresholding techniques"""
        # Adaptive threshold
        adaptive = cv2.adaptiveThreshold(
            img, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            11, 2
        )
        
        # Otsu threshold
        _, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Combine
        combined = cv2.bitwise_and(adaptive, otsu)
        
        return combined
    
    def extract_shape_features(self, img):
        """Trích xuất đặc trưng hình học"""
        # Find contours
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Tính các đặc trưng
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        # Circularity
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Convex hull
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Approximation
        epsilon = 0.04 * perimeter
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        num_vertices = len(approx)
        
        return {
            'circularity': circularity,
            'solidity': solidity,
            'num_vertices': num_vertices,
            'area': area
        }
    
    def detect_shape_type(self, img):
        """Detect loại shape bằng geometric features"""
        features = self.extract_shape_features(img)
        
        if features is None:
            return 'unknown'
        
        # Circle: circularity cao, solidity cao
        if features['circularity'] > 0.8 and features['solidity'] > 0.9:
            return 'circle'
        
        # Square/Rectangle: 4 vertices, solidity cao
        if features['num_vertices'] == 4 and features['solidity'] > 0.85:
            return 'square'
        
        # Triangle: 3 vertices
        if features['num_vertices'] == 3:
            return 'triangle'
        
        # Nếu có nhiều vertices nhỏ -> có thể là chữ số
        if features['num_vertices'] > 6 or features['circularity'] < 0.6:
            return 'digit'
        
        return 'unknown'


class UltraPreprocessor:
    """Preprocessor với tất cả kỹ thuật tiên tiến"""
    
    def __init__(self):
        self.processor = AdvancedImageProcessor()
        self.cache = {}
    
    def preprocess_for_mnist(self, image):
        """Preprocess cho MNIST với kỹ thuật advanced"""
        if isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image.copy()
        
        # Convert to grayscale
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            elif img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Denoise với bilateral filter
        img = self.processor.bilateral_filter(img)
        
        # CLAHE để enhance contrast
        img = self.processor.apply_clahe(img)
        
        # Find và center object
        img = self.processor.find_and_center_object(img)
        
        # Normalize size
        img = self.processor.normalize_size(img, (28, 28))
        
        # Advanced thresholding
        img = self.processor.advanced_threshold(img)
        
        # Morphological operations
        img = self.processor.morphological_operations(img)
        
        # Invert nếu cần (MNIST là chữ trắng nền đen)
        if np.mean(img) > 127:
            img = 255 - img
        
        # Normalize
        img = img.astype('float32') / 255.0
        
        # Reshape
        img = img.reshape(1, 28, 28, 1)
        
        return img
    
    def preprocess_for_shape(self, image):
        """Preprocess cho Shape với kỹ thuật advanced"""
        if isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image.copy()
        
        # Convert to grayscale
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            elif img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Denoise
        img = self.processor.bilateral_filter(img)
        
        # CLAHE
        img = self.processor.apply_clahe(img)
        
        # Find và center object
        img = self.processor.find_and_center_object(img)
        
        # Normalize size
        img = self.processor.normalize_size(img, (64, 64))
        
        # Advanced thresholding
        img = self.processor.advanced_threshold(img)
        
        # Morphological operations
        img = self.processor.morphological_operations(img)
        
        # Edge enhancement
        edges = self.processor.edge_detection_multi(img)
        img = cv2.addWeighted(img, 0.7, edges, 0.3, 0)
        
        # Normalize
        img = img.astype('float32') / 255.0
        
        # Reshape
        img = img.reshape(1, 64, 64, 1)
        
        return img
    
    def auto_detect_and_preprocess(self, image):
        """Auto detect với shape feature analysis"""
        if isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image.copy()
        
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        # Denoise
        gray = self.processor.bilateral_filter(gray)
        
        # Threshold để phân tích
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Detect shape type
        shape_type = self.processor.detect_shape_type(binary)
        
        print(f"Detected type: {shape_type}")
        
        # Preprocess theo loại
        if shape_type in ['circle', 'square', 'triangle']:
            processed = self.preprocess_for_shape(image)
            return processed, 'shape'
        else:
            processed = self.preprocess_for_mnist(image)
            return processed, 'mnist'


_preprocessor = None

def get_preprocessor():
    """Singleton preprocessor"""
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = UltraPreprocessor()
    return _preprocessor


if __name__ == "__main__":
    print("Advanced preprocessor loaded!")