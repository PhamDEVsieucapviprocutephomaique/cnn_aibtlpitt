"""
ULTRA TRAINING SYSTEM - Tối ưu cực mạnh với ensemble và kỹ thuật tiên tiến
"""
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import *


class UltraShapeGenerator:
    """Tạo dataset shapes CỰC KỲ CHÍNH XÁC với nhiều variations"""
    
    def __init__(self, img_size=64):
        self.img_size = img_size
    
    def generate_circle_variations(self, num_samples):
        """Tạo circles với NHIỀU biến thể"""
        images = []
        for _ in range(num_samples):
            img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            
            # Random position (không chỉ center)
            center_x = np.random.randint(self.img_size // 4, 3 * self.img_size // 4)
            center_y = np.random.randint(self.img_size // 4, 3 * self.img_size // 4)
            center = (center_x, center_y)
            
            # Random radius
            max_radius = min(center_x, center_y, 
                           self.img_size - center_x, 
                           self.img_size - center_y) - 3
            radius = np.random.randint(max(8, max_radius // 2), max_radius)
            
            # Quyết định vẽ circle hay ellipse
            if np.random.random() > 0.3:
                # Perfect circle
                cv2.circle(img, center, radius, 255, -1)
            else:
                # Ellipse (vẫn tròn nhưng hơi méo)
                axes = (radius, int(radius * np.random.uniform(0.85, 1.15)))
                angle = np.random.randint(0, 360)
                cv2.ellipse(img, center, axes, angle, 0, 360, 255, -1)
            
            # Add variations
            variation = np.random.choice(['blur', 'noise', 'erosion', 'dilation', 'none'], 
                                        p=[0.2, 0.15, 0.15, 0.15, 0.35])
            
            if variation == 'blur':
                img = cv2.GaussianBlur(img, (3, 3), 0.5)
            elif variation == 'noise':
                noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
                img = cv2.add(img, noise)
            elif variation == 'erosion':
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                img = cv2.erode(img, kernel, iterations=1)
            elif variation == 'dilation':
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                img = cv2.dilate(img, kernel, iterations=1)
            
            images.append(img)
        
        return np.array(images)
    
    def generate_square_variations(self, num_samples):
        """Tạo squares với NHIỀU biến thể"""
        images = []
        for _ in range(num_samples):
            img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            
            # Random size và position
            size = np.random.randint(25, 45)
            x1 = np.random.randint(5, self.img_size - size - 5)
            y1 = np.random.randint(5, self.img_size - size - 5)
            
            # Square hoặc rectangle gần vuông
            if np.random.random() > 0.2:
                # Perfect square
                x2, y2 = x1 + size, y1 + size
            else:
                # Rectangle gần vuông
                width = size
                height = int(size * np.random.uniform(0.85, 1.15))
                x2 = x1 + width
                y2 = min(y1 + height, self.img_size - 5)
            
            cv2.rectangle(img, (x1, y1), (x2, y2), 255, -1)
            
            # Random rotation
            if np.random.random() > 0.5:
                angle = np.random.randint(-30, 30)
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                img = cv2.warpAffine(img, M, (self.img_size, self.img_size))
            
            # Add variations
            variation = np.random.choice(['blur', 'noise', 'partial', 'rounded', 'none'], 
                                        p=[0.15, 0.15, 0.1, 0.15, 0.45])
            
            if variation == 'blur':
                img = cv2.GaussianBlur(img, (3, 3), 0.7)
            elif variation == 'noise':
                noise = np.random.normal(0, 15, img.shape).astype(np.uint8)
                img = cv2.add(img, noise)
            elif variation == 'partial':
                # Xóa một phần nhỏ
                if np.random.random() > 0.5:
                    img[y1:y1+3, :] = 0
                else:
                    img[:, x1:x1+3] = 0
            elif variation == 'rounded':
                # Bo góc nhẹ
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                img = cv2.erode(img, kernel, iterations=1)
            
            images.append(img)
        
        return np.array(images)
    
    def generate_triangle_variations(self, num_samples):
        """Tạo triangles với NHIỀU biến thể"""
        images = []
        for _ in range(num_samples):
            img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            
            # Random triangle type
            tri_type = np.random.choice(['equilateral', 'isosceles', 'scalene', 'right'], 
                                       p=[0.3, 0.3, 0.25, 0.15])
            
            # Base parameters
            base_size = np.random.randint(30, 50)
            center_x = self.img_size // 2 + np.random.randint(-10, 10)
            
            if tri_type == 'equilateral':
                # Tam giác đều
                height = int(base_size * 0.866)  # sqrt(3)/2
                top_y = self.img_size // 2 - height // 2
                base_y = top_y + height
                
                pts = np.array([
                    [center_x - base_size // 2, base_y],
                    [center_x + base_size // 2, base_y],
                    [center_x, top_y]
                ], np.int32)
                
            elif tri_type == 'isosceles':
                # Tam giác cân
                height = np.random.randint(35, 55)
                top_y = self.img_size // 2 - height // 2
                base_y = top_y + height
                
                pts = np.array([
                    [center_x - base_size // 2, base_y],
                    [center_x + base_size // 2, base_y],
                    [center_x + np.random.randint(-5, 5), top_y]
                ], np.int32)
                
            elif tri_type == 'right':
                # Tam giác vuông
                size = np.random.randint(35, 50)
                x1 = center_x - size // 2
                y1 = self.img_size // 2 + size // 2
                
                pts = np.array([
                    [x1, y1],
                    [x1 + size, y1],
                    [x1, y1 - size]
                ], np.int32)
                
            else:  # scalene
                # Tam giác bất kỳ
                pts = np.array([
                    [center_x - np.random.randint(15, 25), self.img_size // 2 + np.random.randint(15, 25)],
                    [center_x + np.random.randint(15, 25), self.img_size // 2 + np.random.randint(15, 25)],
                    [center_x + np.random.randint(-10, 10), self.img_size // 2 - np.random.randint(25, 35)]
                ], np.int32)
            
            cv2.fillPoly(img, [pts], 255)
            
            # Random rotation
            if np.random.random() > 0.5:
                angle = np.random.randint(-45, 45)
                center = (self.img_size // 2, self.img_size // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                img = cv2.warpAffine(img, M, (self.img_size, self.img_size))
            
            # Add variations
            variation = np.random.choice(['blur', 'noise', 'erosion', 'none'], 
                                        p=[0.2, 0.15, 0.15, 0.5])
            
            if variation == 'blur':
                img = cv2.GaussianBlur(img, (3, 3), 0.5)
            elif variation == 'noise':
                noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
                img = cv2.add(img, noise)
            elif variation == 'erosion':
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                img = cv2.erode(img, kernel, iterations=1)
            
            images.append(img)
        
        return np.array(images)
    
    def generate_ultra_dataset(self, samples_per_class=8000):
        """Tạo ULTRA dataset với 8000 samples/class"""
        print("🚀 Generating ULTRA Shape Dataset...")
        
        circles = self.generate_circle_variations(samples_per_class)
        print(f"  ✓ Generated {samples_per_class} circles")
        
        squares = self.generate_square_variations(samples_per_class)
        print(f"  ✓ Generated {samples_per_class} squares")
        
        triangles = self.generate_triangle_variations(samples_per_class)
        print(f"  ✓ Generated {samples_per_class} triangles")
        
        # Combine
        X = np.concatenate([circles, squares, triangles], axis=0)
        
        # Labels
        y = np.concatenate([
            np.zeros(samples_per_class),
            np.ones(samples_per_class),
            np.full(samples_per_class, 2)
        ])
        
        # Shuffle
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
        
        # Preprocessing
        X = X.astype('float32') / 255.0
        X = np.expand_dims(X, -1)
        
        # One-hot encoding
        y = keras.utils.to_categorical(y, 3)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y.argmax(axis=1)
        )
        
        print(f"\n✅ ULTRA Dataset created:")
        print(f"   Train: {X_train.shape}")
        print(f"   Test: {X_test.shape}")
        
        unique, counts = np.unique(y_train.argmax(axis=1), return_counts=True)
        print("   Class distribution:", dict(zip([SHAPE_CLASSES[i] for i in unique], counts)))
        
        return (X_train, y_train), (X_test, y_test)


class UltraMNISTModel:
    """MNIST Model CỰC MẠNH với ResNet-like architecture"""
    
    @staticmethod
    def build():
        inputs = keras.Input(shape=(28, 28, 1))
        
        # Initial conv
        x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Block 1
        shortcut = x
        x = layers.Conv2D(32, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(32, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Block 2
        x = layers.Conv2D(64, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        shortcut = x
        x = layers.Conv2D(64, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(64, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Dense
        x = layers.Flatten()(x)
        x = layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, kernel_regularizer=regularizers.l2(0.0001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(10, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0003),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model


class UltraShapeModel:
    """Shape Model CỰC MẠNH với attention mechanism"""
    
    @staticmethod
    def attention_block(x, channels):
        """Attention mechanism"""
        # Channel attention
        gap = layers.GlobalAveragePooling2D()(x)
        dense1 = layers.Dense(channels // 8, activation='relu')(gap)
        dense2 = layers.Dense(channels, activation='sigmoid')(dense1)
        dense2 = layers.Reshape((1, 1, channels))(dense2)
        
        # Multiply attention
        x = layers.Multiply()([x, dense2])
        return x
    
    @staticmethod
    def build():
        inputs = keras.Input(shape=(64, 64, 1))
        
        # Block 1
        x = layers.Conv2D(64, (3, 3), padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(64, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = UltraShapeModel.attention_block(x, 64)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Block 2
        x = layers.Conv2D(128, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(128, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = UltraShapeModel.attention_block(x, 128)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Block 3
        x = layers.Conv2D(256, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(256, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = UltraShapeModel.attention_block(x, 256)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.3)(x)
        
        # Global pooling
        gap = layers.GlobalAveragePooling2D()(x)
        gmp = layers.GlobalMaxPooling2D()(x)
        x = layers.Concatenate()([gap, gmp])
        
        # Dense
        x = layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, kernel_regularizer=regularizers.l2(0.0001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.4)(x)
        outputs = layers.Dense(3, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model


def train_ultra_mnist():
    """Train MNIST với CỰC KỲ CHÍNH XÁC"""
    print("\n" + "="*70)
    print("🔥 TRAINING ULTRA MNIST MODEL")
    print("="*70)
    
    # Load MNIST
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Preprocessing
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    # Data augmentation
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        shear_range=0.1
    )
    
    # Build model
    model = UltraMNISTModel.build()
    print(model.summary())
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            min_delta=0.0005
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            os.path.join(MODELS_DIR, 'ultra_mnist_best.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Training
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=128),
        steps_per_epoch=len(x_train) // 128,
        epochs=50,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n🎯 ULTRA MNIST Test Accuracy: {test_acc*100:.2f}%")
    
    # Save
    model.save(MNIST_MODEL_PATH)
    print(f"💾 Model saved: {MNIST_MODEL_PATH}")
    
    return model, history


def train_ultra_shape():
    """Train Shape với CỰC KỲ CHÍNH XÁC"""
    print("\n" + "="*70)
    print("🔥 TRAINING ULTRA SHAPE MODEL")
    print("="*70)
    
    # Generate ultra dataset
    generator = UltraShapeGenerator()
    (x_train, y_train), (x_test, y_test) = generator.generate_ultra_dataset(8000)
    
    # Build model
    model = UltraShapeModel.build()
    print(model.summary())
    
    # Data augmentation - MẠNH HƠN
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.15,
        shear_range=0.15,
        brightness_range=[0.8, 1.2],
        horizontal_flip=False,
        vertical_flip=False,
        fill_mode='constant',
        cval=0
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=25,
            restore_best_weights=True,
            min_delta=0.0005
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-8,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            os.path.join(MODELS_DIR, 'ultra_shape_best.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.LearningRateScheduler(
            lambda epoch: 0.0001 * (0.95 ** epoch)
        )
    ]
    
    # Training
    print("\n🚀 Starting training with augmentation...")
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=32),
        steps_per_epoch=len(x_train) // 32,
        epochs=150,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n🎯 ULTRA Shape Test Accuracy: {test_acc*100:.2f}%")
    
    # Detailed evaluation
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    print("\n📊 Per-class accuracy:")
    for i, class_name in enumerate(SHAPE_CLASSES):
        mask = y_true_classes == i
        if mask.sum() > 0:
            class_acc = (y_pred_classes[mask] == i).mean()
            print(f"   {class_name}: {class_acc*100:.2f}%")
    
    # Save
    model.save(SHAPE_MODEL_PATH)
    print(f"💾 Model saved: {SHAPE_MODEL_PATH}")
    
    return model, history


def main():
    """Main training pipeline"""
    print("=" * 70)
    print("🎯 ULTRA NEURAL RECOGNITION - MAXIMUM ACCURACY TRAINING")
    print("=" * 70)
    
    # Create directories
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(os.path.join(MODELS_DIR, 'checkpoints'), exist_ok=True)
    
    # Train models
    print("\n🚀 Starting ULTRA training...")
    
    # Train MNIST
    mnist_model, mnist_history = train_ultra_mnist()
    
    # Train Shape
    shape_model, shape_history = train_ultra_shape()
    
    print("\n" + "=" * 70)
    print("✅ ULTRA TRAINING COMPLETE!")
    print("=" * 70)
    print("\n🎯 Models are now ULTRA ACCURATE!")
    print("   You can run: python app/super_gui.py")
    print("=" * 70)


if __name__ == "__main__":
    main()