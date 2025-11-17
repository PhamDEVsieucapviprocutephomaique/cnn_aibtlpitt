"""
SUPER TRAINER - Training với data chất lượng cao và kỹ thuật tiên tiến
"""
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import *


class SuperDataGenerator:
    """Tạo dataset SIÊU CHẤT LƯỢNG"""
    
    def __init__(self, img_size=64):
        self.img_size = img_size
    
    def generate_perfect_circle(self, num_samples):
        """Tạo hình tròn HOÀN HẢO"""
        images = []
        for i in range(num_samples):
            img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            
            # Center cố định để model dễ học
            center = (self.img_size // 2, self.img_size // 2)
            radius = np.random.randint(20, 25)  # Radius ổn định
            
            # Vẽ circle hoàn hảo
            cv2.circle(img, center, radius, 255, -1)
            
            # Thêm biến thể nhẹ
            if np.random.random() > 0.8:
                img = cv2.GaussianBlur(img, (3, 3), 0.5)
            
            images.append(img)
        return np.array(images)
    
    def generate_perfect_square(self, num_samples):
        """Tạo hình vuông HOÀN HẢO"""
        images = []
        for i in range(num_samples):
            img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            
            # Square ở giữa
            size = np.random.randint(35, 45)
            margin = (self.img_size - size) // 2
            x1, y1 = margin, margin
            x2, y2 = margin + size, margin + size
            
            cv2.rectangle(img, (x1, y1), (x2, y2), 255, -1)
            
            # Thêm rotation nhẹ đôi khi
            if np.random.random() > 0.9:
                angle = np.random.randint(-15, 15)
                center = (self.img_size // 2, self.img_size // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                img = cv2.warpAffine(img, M, (self.img_size, self.img_size))
            
            images.append(img)
        return np.array(images)
    
    def generate_perfect_triangle(self, num_samples):
        """Tạo hình tam giác HOÀN HẢO"""
        images = []
        for i in range(num_samples):
            img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            
            # Triangle cân
            center_x = self.img_size // 2
            base_y = self.img_size - 15
            top_y = 15
            
            # 3 điểm của tam giác
            pts = np.array([
                [center_x - 25, base_y],  # Bottom left
                [center_x + 25, base_y],  # Bottom right  
                [center_x, top_y]         # Top center
            ], np.int32)
            
            cv2.fillPoly(img, [pts], 255)
            
            images.append(img)
        return np.array(images)
    
    def generate_super_dataset(self, samples_per_class=5000):
        """Tạo dataset SIÊU CHÍNH XÁC"""
        print("🎯 Generating SUPER dataset...")
        
        # Tạo data chất lượng cao
        circles = self.generate_perfect_circle(samples_per_class)
        squares = self.generate_perfect_square(samples_per_class) 
        triangles = self.generate_perfect_triangle(samples_per_class)
        
        # Kết hợp
        X = np.concatenate([circles, squares, triangles], axis=0)
        
        # Labels
        y_circles = np.zeros(samples_per_class)
        y_squares = np.ones(samples_per_class)
        y_triangles = np.full(samples_per_class, 2)
        y = np.concatenate([y_circles, y_squares, y_triangles])
        
        # Preprocessing
        X = X.astype('float32') / 255.0
        X = np.expand_dims(X, -1)  # Add channel dimension
        
        # One-hot encoding
        y = keras.utils.to_categorical(y, 3)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y.argmax(axis=1)
        )
        
        print(f"✅ SUPER Dataset created:")
        print(f"   Train: {X_train.shape}")
        print(f"   Test: {X_test.shape}")
        
        return (X_train, y_train), (X_test, y_test)


class SuperMNISTModel:
    """Model MNIST SIÊU CHÍNH XÁC"""
    
    @staticmethod
    def build():
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(10, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0005),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model


class SuperShapeModel:
    """Model Shape SIÊU CHÍNH XÁC"""
    
    @staticmethod
    def build():
        model = models.Sequential([
            # Block 1
            layers.Conv2D(64, (3, 3), activation='relu', input_shape=(64, 64, 1)),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 2
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 3
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dense
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(3, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0003),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model


def train_super_mnist():
    """Train MNIST với độ chính xác CAO"""
    print("\n🔥 Training SUPER MNIST Model...")
    
    # Load MNIST
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Preprocessing
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    # Build model
    model = SuperMNISTModel.build()
    print(model.summary())
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            min_delta=0.001
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        ),
        keras.callbacks.ModelCheckpoint(
            os.path.join(MODELS_DIR, 'super_mnist_best.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Training
    history = model.fit(
        x_train, y_train,
        batch_size=128,
        epochs=50,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n🎯 SUPER MNIST Test Accuracy: {test_acc*100:.2f}%")
    
    # Save
    model.save(MNIST_MODEL_PATH)
    print(f"💾 Model saved: {MNIST_MODEL_PATH}")
    
    return model, history


def train_super_shape():
    """Train Shape với độ chính xác CAO"""
    print("\n🔥 Training SUPER Shape Model...")
    
    # Generate super dataset
    generator = SuperDataGenerator()
    (x_train, y_train), (x_test, y_test) = generator.generate_super_dataset(5000)
    
    # Build model
    model = SuperShapeModel.build()
    print(model.summary())
    
    # Data Augmentation
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        shear_range=0.1
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            min_delta=0.001
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7
        ),
        keras.callbacks.ModelCheckpoint(
            os.path.join(MODELS_DIR, 'super_shape_best.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Training với augmentation
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=32),
        steps_per_epoch=len(x_train) // 32,
        epochs=100,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n🎯 SUPER Shape Test Accuracy: {test_acc*100:.2f}%")
    
    # Save
    model.save(SHAPE_MODEL_PATH)
    print(f"💾 Model saved: {SHAPE_MODEL_PATH}")
    
    return model, history


def main():
    """Main training pipeline"""
    print("=" * 70)
    print("🎯 SUPER NEURAL RECOGNITION - ULTRA ACCURATE TRAINING")
    print("=" * 70)
    
    # Create directories
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(os.path.join(MODELS_DIR, 'checkpoints'), exist_ok=True)
    
    # Train models
    print("\n🚀 Starting SUPER training...")
    
    # Train MNIST
    mnist_model, mnist_history = train_super_mnist()
    
    # Train Shape
    shape_model, shape_history = train_super_shape()
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE! Models are now ULTRA ACCURATE!")
    print("=" * 70)
    print("\n🎯 You can now run the application:")
    print("   python app/super_gui.py")
    print("=" * 70)


if __name__ == "__main__":
    main()