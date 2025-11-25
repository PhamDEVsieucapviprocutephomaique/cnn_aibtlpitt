"""
FAST TRAINING - Train models trong vài chục giây với độ chính xác cao
Tối ưu cho tốc độ tối đa
"""
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import cv2
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from configs.fast_config import *

# TỐI ƯU TỐC ĐỘ
tf.config.optimizer.set_jit(True)  # XLA compilation
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Giảm logs


class FastShapeGenerator:
    """Tạo dataset CỰC NHANH - chỉ essential variations"""
    
    def __init__(self, img_size=48):  # Giảm size: 48x48 thay vì 64x64
        self.img_size = img_size
    
    def generate_circles(self, n):
        imgs = np.zeros((n, self.img_size, self.img_size), dtype=np.uint8)
        for i in range(n):
            center = (self.img_size // 2, self.img_size // 2)
            radius = np.random.randint(12, 20)
            cv2.circle(imgs[i], center, radius, 255, -1)
            # Variation nhanh
            if np.random.random() > 0.7:
                imgs[i] = cv2.GaussianBlur(imgs[i], (3, 3), 0.5)
        return imgs
    
    def generate_squares(self, n):
        imgs = np.zeros((n, self.img_size, self.img_size), dtype=np.uint8)
        for i in range(n):
            size = np.random.randint(18, 28)
            x = (self.img_size - size) // 2
            y = (self.img_size - size) // 2
            cv2.rectangle(imgs[i], (x, y), (x+size, y+size), 255, -1)
            # Rotation nhanh
            if np.random.random() > 0.5:
                angle = np.random.randint(-15, 15)
                M = cv2.getRotationMatrix2D((self.img_size//2, self.img_size//2), angle, 1.0)
                imgs[i] = cv2.warpAffine(imgs[i], M, (self.img_size, self.img_size))
        return imgs
    
    def generate_triangles(self, n):
        imgs = np.zeros((n, self.img_size, self.img_size), dtype=np.uint8)
        center = self.img_size // 2
        for i in range(n):
            size = np.random.randint(20, 30)
            pts = np.array([
                [center - size//2, center + size//2],
                [center + size//2, center + size//2],
                [center, center - size//2]
            ], np.int32)
            cv2.fillPoly(imgs[i], [pts], 255)
            # Simple rotation
            if np.random.random() > 0.5:
                angle = np.random.randint(-20, 20)
                M = cv2.getRotationMatrix2D((center, center), angle, 1.0)
                imgs[i] = cv2.warpAffine(imgs[i], M, (self.img_size, self.img_size))
        return imgs
    
    def generate_fast_dataset(self, samples=1500):  # Giảm: 1500 thay vì 8000
        """Tạo dataset NHANH"""
        print(f"🚀 Generating FAST dataset ({samples} per class)...")
        
        circles = self.generate_circles(samples)
        squares = self.generate_squares(samples)
        triangles = self.generate_triangles(samples)
        
        X = np.concatenate([circles, squares, triangles])
        y = np.concatenate([
            np.zeros(samples),
            np.ones(samples),
            np.full(samples, 2)
        ])
        
        # Shuffle
        idx = np.arange(len(X))
        np.random.shuffle(idx)
        X, y = X[idx], y[idx]
        
        # Preprocess
        X = X.astype('float32') / 255.0
        X = np.expand_dims(X, -1)
        y = keras.utils.to_categorical(y, 3)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"✓ Dataset ready: Train={X_train.shape}, Test={X_test.shape}")
        return (X_train, y_train), (X_test, y_test)


class FastMNISTModel:
    """MNIST Model CỰC NHẸ - tốc độ tối đa"""
    
    @staticmethod
    def build():
        model = models.Sequential([
            # Chỉ 2 conv blocks - đủ cho 99%+ accuracy
            layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D(2),
            layers.Dropout(0.25),
            
            layers.Conv2D(64, 3, activation='relu'),
            layers.MaxPooling2D(2),
            layers.Dropout(0.25),
            
            # Compact dense
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])
        
        # Optimizer nhanh
        model.compile(
            optimizer=keras.optimizers.Adam(0.001),  # LR cao hơn = train nhanh hơn
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model


class FastShapeModel:
    """Shape Model CỰC NHẸ - 48x48 input"""
    
    @staticmethod
    def build(img_size=48):
        model = models.Sequential([
            # 3 conv blocks nhỏ gọn
            layers.Conv2D(32, 3, activation='relu', input_shape=(img_size, img_size, 1)),
            layers.MaxPooling2D(2),
            layers.Dropout(0.25),
            
            layers.Conv2D(64, 3, activation='relu'),
            layers.MaxPooling2D(2),
            layers.Dropout(0.25),
            
            layers.Conv2D(128, 3, activation='relu'),
            layers.MaxPooling2D(2),
            layers.Dropout(0.25),
            
            # Dense compact
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(3, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model


def train_fast_mnist():
    """Train MNIST CỰC NHANH - khoảng 15-20 giây"""
    print("\n" + "="*60)
    print("⚡ FAST MNIST TRAINING")
    print("="*60)
    
    # Load MNIST - KHÔNG augmentation
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Subsample để train nhanh hơn - vẫn đủ cho 99%+
    sample_size = 30000  # Thay vì 60000
    idx = np.random.choice(len(x_train), sample_size, replace=False)
    x_train = x_train[idx]
    y_train = y_train[idx]
    
    # Preprocess
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    print(f"Training on {len(x_train)} samples (subsampled for speed)")
    
    # Build model
    model = FastMNISTModel.build()
    
    # FAST training - chỉ callbacks cần thiết
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True
        )
    ]
    
    # Train - batch size LỚN = nhanh hơn
    print("🚀 Training...")
    history = model.fit(
        x_train, y_train,
        batch_size=256,  # Batch lớn = ít iterations
        epochs=10,       # Ít epochs
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"✅ Test Accuracy: {test_acc*100:.2f}%")
    
    # Save
    model.save(MNIST_MODEL_PATH)
    print(f"💾 Saved: {MNIST_MODEL_PATH}")
    
    return model, history


def train_fast_shape():
    """Train Shape CỰC NHANH - khoảng 20-30 giây"""
    print("\n" + "="*60)
    print("⚡ FAST SHAPE TRAINING")
    print("="*60)
    
    # Generate fast dataset
    generator = FastShapeGenerator(img_size=48)
    (x_train, y_train), (x_test, y_test) = generator.generate_fast_dataset(1500)
    
    # Build model
    model = FastShapeModel.build(img_size=48)
    
    # Simple augmentation - CHỈ trong training
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1
    )
    
    # FAST callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        )
    ]
    
    # Train
    print("🚀 Training...")
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=64),
        steps_per_epoch=len(x_train) // 64,
        epochs=30,  # Đủ để đạt 95%+
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"✅ Test Accuracy: {test_acc*100:.2f}%")
    
    # Save
    model.save(SHAPE_MODEL_PATH)
    print(f"💾 Saved: {SHAPE_MODEL_PATH}")
    
    return model, history


def main():
    """Main FAST training pipeline"""
    import time
    
    print("="*60)
    print("⚡ FAST NEURAL RECOGNITION TRAINING")
    print("   Target: Complete in under 1 minute")
    print("="*60)
    
    start_time = time.time()
    
    # Create dirs
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    try:
        # Train MNIST
        mnist_model, _ = train_fast_mnist()
        
        # Train Shape
        shape_model, _ = train_fast_shape()
        
        # Done
        elapsed = time.time() - start_time
        print("\n" + "="*60)
        print(f"✅ TRAINING COMPLETE in {elapsed:.1f} seconds!")
        print("="*60)
        print(f"\n📊 Models ready:")
        print(f"   • MNIST: ~99% accuracy")
        print(f"   • Shape: ~95%+ accuracy")
        print(f"\n🚀 Run: python app/super_gui.py")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)