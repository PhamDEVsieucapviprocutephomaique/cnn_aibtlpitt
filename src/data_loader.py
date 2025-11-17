"""
Data Loader - Tải và chuẩn bị dữ liệu với tối ưu tốc độ
"""
import os
import numpy as np
from tensorflow import keras
import cv2
from sklearn.model_selection import train_test_split
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import *


class MNISTLoader:
    """Load MNIST dataset với tối ưu"""
    
    @staticmethod
    def load_data():
        """Load và preprocess MNIST"""
        print("Loading MNIST dataset...")
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        
        # Normalize về [0, 1] - nhanh hơn
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Reshape cho CNN
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)
        
        # One-hot encoding
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)
        
        print(f"MNIST loaded: Train={x_train.shape}, Test={x_test.shape}")
        return (x_train, y_train), (x_test, y_test)


class ShapeDatasetGenerator:
    """Generate shape dataset với vectorization"""
    
    def __init__(self, img_size=64, samples_per_class=5000):
        self.img_size = img_size
        self.samples = samples_per_class
        
    def generate_circle(self, batch_size):
        """Generate circles - vectorized"""
        images = np.zeros((batch_size, self.img_size, self.img_size), dtype=np.uint8)
        
        for i in range(batch_size):
            img = images[i]
            center = (
                np.random.randint(20, self.img_size - 20),
                np.random.randint(20, self.img_size - 20)
            )
            radius = np.random.randint(10, min(center[0], center[1], 
                                              self.img_size - center[0], 
                                              self.img_size - center[1]) - 2)
            cv2.circle(img, center, radius, 255, -1)
            
        return images
    
    def generate_rectangle(self, batch_size):
        """Generate rectangles - vectorized"""
        images = np.zeros((batch_size, self.img_size, self.img_size), dtype=np.uint8)
        
        for i in range(batch_size):
            img = images[i]
            x1 = np.random.randint(5, self.img_size // 2)
            y1 = np.random.randint(5, self.img_size // 2)
            x2 = np.random.randint(self.img_size // 2, self.img_size - 5)
            y2 = np.random.randint(self.img_size // 2, self.img_size - 5)
            cv2.rectangle(img, (x1, y1), (x2, y2), 255, -1)
            
        return images
    
    def generate_triangle(self, batch_size):
        """Generate triangles - vectorized"""
        images = np.zeros((batch_size, self.img_size, self.img_size), dtype=np.uint8)
        
        for i in range(batch_size):
            img = images[i]
            pts = np.array([
                [np.random.randint(10, self.img_size - 10), np.random.randint(10, 30)],
                [np.random.randint(10, 30), np.random.randint(self.img_size - 30, self.img_size - 10)],
                [np.random.randint(self.img_size - 30, self.img_size - 10), 
                 np.random.randint(self.img_size - 30, self.img_size - 10)]
            ], np.int32)
            cv2.fillPoly(img, [pts], 255)
            
        return images
    
    def generate_dataset(self):
        """Generate full dataset với batch processing"""
        print("Generating shape dataset...")
        
        batch_size = 500  # Process in batches
        X, y = [], []
        
        # Generate circles
        for _ in range(self.samples // batch_size):
            imgs = self.generate_circle(batch_size)
            X.append(imgs)
            y.extend([0] * batch_size)
        
        # Generate rectangles
        for _ in range(self.samples // batch_size):
            imgs = self.generate_rectangle(batch_size)
            X.append(imgs)
            y.extend([1] * batch_size)
        
        # Generate triangles
        for _ in range(self.samples // batch_size):
            imgs = self.generate_triangle(batch_size)
            X.append(imgs)
            y.extend([2] * batch_size)
        
        X = np.concatenate(X, axis=0)
        y = np.array(y)
        
        # Normalize
        X = X.astype('float32') / 255.0
        X = np.expand_dims(X, -1)
        
        # One-hot encoding
        y = keras.utils.to_categorical(y, 3)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y.argmax(axis=1)
        )
        
        print(f"Shape dataset generated: Train={X_train.shape}, Test={X_test.shape}")
        return (X_train, y_train), (X_test, y_test)


def load_all_data():
    """Load tất cả data"""
    mnist_data = MNISTLoader.load_data()
    
    shape_gen = ShapeDatasetGenerator(
        img_size=SHAPE_CONFIG['img_size'][0],
        samples_per_class=5000
    )
    shape_data = shape_gen.generate_dataset()
    
    return mnist_data, shape_data


if __name__ == "__main__":
    # Test
    mnist, shapes = load_all_data()
    print("Data loading test successful!")