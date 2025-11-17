"""
Image Preprocessor - ĐÃ FIX: bỏ inversion cho shape
"""
import cv2
import numpy as np
from PIL import Image
import hashlib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import *


class ImageCache:
    """LRU Cache cho ảnh đã xử lý"""
    
    def __init__(self, max_size=100):
        self.cache = {}
        self.access_order = []
        self.max_size = max_size
    
    def get_hash(self, img_array):
        """Tạo hash từ ảnh"""
        return hashlib.md5(img_array.tobytes()).hexdigest()
    
    def get(self, img_array):
        """Lấy từ cache"""
        img_hash = self.get_hash(img_array)
        if img_hash in self.cache:
            self.access_order.remove(img_hash)
            self.access_order.append(img_hash)
            return self.cache[img_hash]
        return None
    
    def put(self, img_array, processed):
        """Thêm vào cache"""
        img_hash = self.get_hash(img_array)
        
        if len(self.cache) >= self.max_size:
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
        
        self.cache[img_hash] = processed
        self.access_order.append(img_hash)


class FixedPreprocessor:
    """Preprocessor ĐÃ FIX: shape giữ nguyên, mnist mới invert"""
    
    def __init__(self):
        self.cache = ImageCache(PREPROCESS_CONFIG['cache_size']) if PREPROCESS_CONFIG['use_cache'] else None
    
    def preprocess_for_mnist(self, image, use_cache=True):
        """
        Preprocess ảnh cho MNIST - CÓ INVERT (vì dataset MNIST là chữ trắng nền đen)
        """
        if isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image.copy()
        
        if use_cache and self.cache:
            cached = self.cache.get(img)
            if cached is not None:
                return cached
        
        # Convert to grayscale
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            elif img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Resize
        img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        
        # CHỈ MNIST MỚI INVERT - QUAN TRỌNG!
        if np.mean(img) > 127:  # Nếu nền trắng
            img = 255 - img     # Đảo thành nền đen chữ trắng
        
        # Normalize
        img = img.astype('float32') / 255.0
        
        img = img.reshape(1, 28, 28, 1)
        
        if use_cache and self.cache:
            self.cache.put(image if isinstance(image, np.ndarray) else np.array(image), img)
        
        return img
    
    def preprocess_for_shape(self, image, use_cache=True):
        """
        Preprocess ảnh cho Shape detection - KHÔNG INVERT (giữ nguyên bản vẽ)
        """
        if isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image.copy()
        
        if use_cache and self.cache:
            cached = self.cache.get(img)
            if cached is not None:
                return cached
        
        # Convert to grayscale
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            elif img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Resize
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
        
        # KHÔNG THRESHOLD, KHÔNG INVERT - GIỮ NGUYÊN BẢN VẼ
        
        # Normalize
        img = img.astype('float32') / 255.0
        
        img = img.reshape(1, 64, 64, 1)
        
        if use_cache and self.cache:
            self.cache.put(image if isinstance(image, np.ndarray) else np.array(image), img)
        
        return img
    
    def auto_detect_and_preprocess(self, image):
        """
        Auto detect loại ảnh bằng diện tích contour
        Returns: (processed_img, detected_type)
        """
        if isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image.copy()
        
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        # Phân tích contour để phân biệt số vs hình
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(max_contour)
            img_area = gray.shape[0] * gray.shape[1]
            
            # Hình lớn -> shape, số nhỏ -> mnist
            if area / img_area > 0.3:  # Chiếm >30% diện tích
                processed = self.preprocess_for_shape(image)
                return processed, 'shape'
        
        # Mặc định là số
        processed = self.preprocess_for_mnist(image)
        return processed, 'mnist'


_preprocessor = None

def get_preprocessor():
    """Singleton preprocessor"""
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = FixedPreprocessor()
    return _preprocessor