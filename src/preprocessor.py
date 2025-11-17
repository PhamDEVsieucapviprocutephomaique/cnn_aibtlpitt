"""
Image Preprocessor - Xử lý ảnh siêu nhanh với caching và vectorization
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


class FastPreprocessor:
    """Preprocessor tối ưu tốc độ"""
    
    def __init__(self):
        self.cache = ImageCache(PREPROCESS_CONFIG['cache_size']) if PREPROCESS_CONFIG['use_cache'] else None
    
    def preprocess_for_mnist(self, image, use_cache=True):
        """
        Preprocess ảnh cho MNIST - siêu tối ưu
        Input: PIL Image hoặc numpy array
        Output: numpy array (1, 28, 28, 1)
        """
        if isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image
        
        if use_cache and self.cache:
            cached = self.cache.get(img)
            if cached is not None:
                return cached
        
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            elif img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        
        if np.mean(img) > 127:
            img = 255 - img
        
        img = img.astype('float32') / 255.0
        
        img = img.reshape(1, 28, 28, 1)
        
        if use_cache and self.cache:
            self.cache.put(image if isinstance(image, np.ndarray) else np.array(image), img)
        
        return img
    
    def preprocess_for_shape(self, image, use_cache=True):
        """
        Preprocess ảnh cho Shape detection - siêu tối ưu
        Input: PIL Image hoặc numpy array
        Output: numpy array (1, 64, 64, 1)
        """
        if isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image
        
        if use_cache and self.cache:
            cached = self.cache.get(img)
            if cached is not None:
                return cached
        
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            elif img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
        
        img = cv2.bilateralFilter(img, 5, 50, 50)
        
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        img = img.astype('float32') / 255.0
        
        img = img.reshape(1, 64, 64, 1)
        
        if use_cache and self.cache:
            self.cache.put(image if isinstance(image, np.ndarray) else np.array(image), img)
        
        return img
    
    def auto_detect_and_preprocess(self, image):
        """
        Auto detect loại ảnh và preprocess phù hợp
        Returns: (processed_img, detected_type)
        """
        if isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image
        
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(max_contour)
            img_area = gray.shape[0] * gray.shape[1]
            
            if area / img_area > 0.1:
                processed = self.preprocess_for_shape(image)
                return processed, 'shape'
        
        processed = self.preprocess_for_mnist(image)
        return processed, 'mnist'
    
    def batch_preprocess(self, images, target='mnist'):
        """Preprocess batch ảnh - vectorized"""
        if target == 'mnist':
            return np.concatenate([self.preprocess_for_mnist(img, use_cache=False) for img in images])
        else:
            return np.concatenate([self.preprocess_for_shape(img, use_cache=False) for img in images])


_preprocessor = None

def get_preprocessor():
    """Singleton preprocessor"""
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = FastPreprocessor()
    return _preprocessor


if __name__ == "__main__":
    preprocessor = FastPreprocessor()
    
    test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    import time
    
    start = time.perf_counter()
    for _ in range(100):
        result = preprocessor.preprocess_for_mnist(test_img)
    elapsed = (time.perf_counter() - start) * 1000 / 100
    print(f"MNIST preprocessing: {elapsed:.2f}ms per image")
    
    start = time.perf_counter()
    for _ in range(100):
        result = preprocessor.preprocess_for_shape(test_img)
    elapsed = (time.perf_counter() - start) * 1000 / 100
    print(f"Shape preprocessing: {elapsed:.2f}ms per image")
    
    print("\nPreprocessor test successful!")