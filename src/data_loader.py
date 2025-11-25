"""
Enhanced Data Loader - Better shape generation
"""
import os
import numpy as np
from tensorflow import keras
import cv2
from sklearn.model_selection import train_test_split
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.fast_config import *


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


class EnhancedShapeDatasetGenerator:
    """Generate better shape dataset với variations"""
    
    def __init__(self, img_size=64, samples_per_class=3000):
        self.img_size = img_size
        self.samples = samples_per_class
    
    def generate_circle(self, batch_size):
        """Generate circles với variations"""
        images = np.zeros((batch_size, self.img_size, self.img_size), dtype=np.uint8)
        
        for i in range(batch_size):
            img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            
            # Random center và radius
            center = (
                np.random.randint(15, self.img_size - 15),
                np.random.randint(15, self.img_size - 15)
            )
            max_radius = min(center[0], center[1], 
                           self.img_size - center[0], 
                           self.img_size - center[1]) - 5
            radius = np.random.randint(8, max_radius)
            
            # Draw circle
            cv2.circle(img, center, radius, 255, -1)
            
            # Add some noise/variations
            if np.random.random() > 0.7:
                # Make ellipse sometimes
                axes = (radius, int(radius * np.random.uniform(0.8, 1.2)))
                angle = np.random.randint(0, 180)
                cv2.ellipse(img, center, axes, angle, 0, 360, 255, -1)
            
            images[i] = img
            
        return images
    
    def generate_square(self, batch_size):
        """Generate squares với variations"""
        images = np.zeros((batch_size, self.img_size, self.img_size), dtype=np.uint8)
        
        for i in range(batch_size):
            img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            
            # Random position và size
            size = np.random.randint(20, 40)
            x1 = np.random.randint(5, self.img_size - size - 5)
            y1 = np.random.randint(5, self.img_size - size - 5)
            x2 = x1 + size
            y2 = y1 + size
            
            # Draw square/rectangle
            if np.random.random() > 0.3:
                # Perfect square
                cv2.rectangle(img, (x1, y1), (x2, y2), 255, -1)
            else:
                # Rectangle với tỷ lệ gần vuông
                width = size
                height = int(size * np.random.uniform(0.8, 1.2))
                x2 = x1 + width
                y2 = y1 + height
                cv2.rectangle(img, (x1, y1), (x2, y2), 255, -1)
            
            # Rotate sometimes
            if np.random.random() > 0.8:
                angle = np.random.randint(0, 90)
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                img = cv2.warpAffine(img, rotation_matrix, (self.img_size, self.img_size))
            
            images[i] = img
            
        return images
    
    def generate_triangle(self, batch_size):
        """Generate triangles với variations"""
        images = np.zeros((batch_size, self.img_size, self.img_size), dtype=np.uint8)
        
        for i in range(batch_size):
            img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            
            # Random triangle points
            base_y = np.random.randint(40, self.img_size - 10)
            top_y = np.random.randint(10, 30)
            
            # Different triangle types
            triangle_type = np.random.choice(['equilateral', 'isosceles', 'scalene'])
            
            if triangle_type == 'equilateral':
                base_length = np.random.randint(25, 40)
                x_center = np.random.randint(20, self.img_size - 20)
                x1 = x_center - base_length // 2
                x2 = x_center + base_length // 2
                x3 = x_center
                y1 = base_y
                y2 = base_y
                y3 = top_y
                
            elif triangle_type == 'isosceles':
                base_length = np.random.randint(25, 45)
                x_center = np.random.randint(20, self.img_size - 20)
                x1 = x_center - base_length // 2
                x2 = x_center + base_length // 2
                x3 = x_center + np.random.randint(-10, 10)
                y1 = base_y
                y2 = base_y
                y3 = top_y
                
            else:  # scalene
                x1 = np.random.randint(10, self.img_size // 2)
                x2 = np.random.randint(self.img_size // 2, self.img_size - 10)
                x3 = np.random.randint(15, self.img_size - 15)
                y1 = base_y
                y2 = base_y - np.random.randint(0, 10)
                y3 = top_y
            
            pts = np.array([[x1, y1], [x2, y2], [x3, y3]], np.int32)
            cv2.fillPoly(img, [pts], 255)
            
            images[i] = img
            
        return images
    
    def generate_negative_samples(self, batch_size):
        """Generate negative samples (random noise, lines, etc.)"""
        images = np.zeros((batch_size, self.img_size, self.img_size), dtype=np.uint8)
        
        for i in range(batch_size):
            img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            
            pattern_type = np.random.choice(['noise', 'lines', 'mixed'])
            
            if pattern_type == 'noise':
                # Random noise
                noise = np.random.randint(0, 100, (self.img_size, self.img_size))
                img = np.where(noise > 95, 255, 0).astype(np.uint8)
                
            elif pattern_type == 'lines':
                # Random lines
                for _ in range(np.random.randint(3, 8)):
                    x1, y1 = np.random.randint(0, self.img_size, 2)
                    x2, y2 = np.random.randint(0, self.img_size, 2)
                    cv2.line(img, (x1, y1), (x2, y2), 255, 2)
                    
            else:  # mixed
                # Combination
                for _ in range(np.random.randint(2, 5)):
                    x1, y1 = np.random.randint(0, self.img_size, 2)
                    x2, y2 = np.random.randint(0, self.img_size, 2)
                    cv2.line(img, (x1, y1), (x2, y2), 255, 1)
                
                # Add some noise
                noise_mask = np.random.random((self.img_size, self.img_size)) > 0.95
                img[noise_mask] = 255
            
            images[i] = img
            
        return images
    
    def generate_dataset(self):
        """Generate enhanced dataset"""
        print("Generating enhanced shape dataset...")
        
        batch_size = 300
        X, y = [], []
        
        # Generate circles
        print("Generating circles...")
        for _ in range(self.samples // batch_size):
            imgs = self.generate_circle(batch_size)
            X.append(imgs)
            y.extend([0] * batch_size)
        
        # Generate squares
        print("Generating squares...")
        for _ in range(self.samples // batch_size):
            imgs = self.generate_square(batch_size)
            X.append(imgs)
            y.extend([1] * batch_size)
        
        # Generate triangles
        print("Generating triangles...")
        for _ in range(self.samples // batch_size):
            imgs = self.generate_triangle(batch_size)
            X.append(imgs)
            y.extend([2] * batch_size)
        
        # Add some negative samples
        print("Adding negative samples...")
        negative_samples = self.samples // 6
        for _ in range(negative_samples // batch_size):
            imgs = self.generate_negative_samples(batch_size)
            X.append(imgs)
            # Negative samples get distributed across classes
            y.extend([np.random.randint(0, 3) for _ in range(batch_size)])
        
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
        
        print(f"Enhanced shape dataset generated:")
        print(f"  Train: {X_train.shape}")
        print(f"  Test: {X_test.shape}")
        
        # Print class distribution
        unique, counts = np.unique(y_train.argmax(axis=1), return_counts=True)
        print("  Class distribution:", dict(zip([SHAPE_CLASSES[i] for i in unique], counts)))
        
        return (X_train, y_train), (X_test, y_test)


def load_all_data():
    """Load tất cả data"""
    mnist_data = MNISTLoader.load_data()
    
    shape_gen = EnhancedShapeDatasetGenerator(
        img_size=SHAPE_CONFIG['img_size'][0],
        samples_per_class=3000
    )
    shape_data = shape_gen.generate_dataset()
    
    return mnist_data, shape_data


if __name__ == "__main__":
    # Test dataset generation
    mnist, shapes = load_all_data()
    print("Enhanced data loading test successful!")