"""
FAST TRAINING SCRIPT - Train models trong vài chục giây
Chạy file này để train cả 2 models nhanh chóng
"""
import os
import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import cv2
from sklearn.model_selection import train_test_split

# Tối ưu TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.optimizer.set_jit(True)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
MNIST_MODEL_PATH = os.path.join(MODELS_DIR, 'mnist_model.h5')
SHAPE_MODEL_PATH = os.path.join(MODELS_DIR, 'shape_model.h5')


# ============================================================================
# SHAPE DATASET GENERATOR
# ============================================================================
class FastShapeGenerator:
    """Tạo shape dataset CỰC NHANH"""
    
    def __init__(self, img_size=48):
        self.img_size = img_size
    
    def generate_circles(self, n):
        imgs = np.zeros((n, self.img_size, self.img_size), dtype=np.uint8)
        for i in range(n):
            center = (self.img_size // 2, self.img_size // 2)
            radius = np.random.randint(12, 20)
            cv2.circle(imgs[i], center, radius, 255, -1)
            # Thêm variation
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
            # Rotation
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
            # Rotation
            if np.random.random() > 0.5:
                angle = np.random.randint(-20, 20)
                M = cv2.getRotationMatrix2D((center, center), angle, 1.0)
                imgs[i] = cv2.warpAffine(imgs[i], M, (self.img_size, self.img_size))
        return imgs
    
    def generate_dataset(self, samples_per_class=1500):
        """Generate dataset"""
        print(f"   Generating {samples_per_class} samples per class...")
        
        circles = self.generate_circles(samples_per_class)
        squares = self.generate_squares(samples_per_class)
        triangles = self.generate_triangles(samples_per_class)
        
        X = np.concatenate([circles, squares, triangles])
        y = np.concatenate([
            np.zeros(samples_per_class),
            np.ones(samples_per_class),
            np.full(samples_per_class, 2)
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
        
        print(f"   Dataset ready: {X_train.shape[0]} train, {X_test.shape[0]} test")
        return (X_train, y_train), (X_test, y_test)


# ============================================================================
# MODEL BUILDERS - IMPROVED cho digits
# ============================================================================
def build_mnist_model():
    """Build MNIST model với architecture tốt hơn"""
    model = models.Sequential([
        # Conv block 1
        layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),
        
        # Conv block 2
        layers.Conv2D(64, 3, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),
        
        # Conv block 3 - THÊM để nhận diện tốt hơn
        layers.Conv2D(128, 3, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_shape_model():
    """Build Shape model"""
    model = models.Sequential([
        layers.Conv2D(32, 3, activation='relu', input_shape=(48, 48, 1)),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),
        
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),
        
        layers.Conv2D(128, 3, activation='relu'),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),
        
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


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================
def train_mnist():
    """Train MNIST model"""
    print("\n" + "="*70)
    print("⚡ TRAINING MNIST MODEL")
    print("="*70)
    
    start_time = time.time()
    
    # Load MNIST
    print("   Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Subsample nhưng GIỮ ĐỦ samples để accuracy cao
    sample_size = 40000  # Tăng từ 30000 -> 40000
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
    
    print(f"   Training on {len(x_train)} samples")
    
    # Build model
    print("   Building model...")
    model = build_mnist_model()
    
    # Data augmentation - ÍT ĐỦ để tăng robustness
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,  # Tăng patience
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
    ]
    
    # Train
    print("   Training...")
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=128),
        steps_per_epoch=len(x_train) // 128,
        epochs=15,  # Tăng epochs
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=2
    )
    
    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    
    # Save
    model.save(MNIST_MODEL_PATH)
    
    elapsed = time.time() - start_time
    
    print(f"\n✅ MNIST COMPLETE in {elapsed:.1f}s")
    print(f"   Test Accuracy: {test_acc*100:.2f}%")
    print(f"   Saved: {MNIST_MODEL_PATH}")
    
    return model, test_acc, elapsed


def train_shape():
    """Train Shape model"""
    print("\n" + "="*70)
    print("⚡ TRAINING SHAPE MODEL")
    print("="*70)
    
    start_time = time.time()
    
    # Generate dataset
    print("   Generating shape dataset...")
    generator = FastShapeGenerator(img_size=48)
    (x_train, y_train), (x_test, y_test) = generator.generate_dataset(1500)
    
    # Build model
    print("   Building model...")
    model = build_shape_model()
    
    # Augmentation
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        )
    ]
    
    # Train
    print("   Training...")
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=64),
        steps_per_epoch=len(x_train) // 64,
        epochs=30,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=2
    )
    
    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    
    # Save
    model.save(SHAPE_MODEL_PATH)
    
    elapsed = time.time() - start_time
    
    print(f"\n✅ SHAPE COMPLETE in {elapsed:.1f}s")
    print(f"   Test Accuracy: {test_acc*100:.2f}%")
    print(f"   Saved: {SHAPE_MODEL_PATH}")
    
    return model, test_acc, elapsed


# ============================================================================
# MAIN
# ============================================================================
def main():
    """Main training pipeline"""
    print("\n" + "="*70)
    print("⚡ FAST NEURAL RECOGNITION - QUICK TRAINING")
    print("="*70)
    print("\n🎯 Target: Train in ~1 minute with high accuracy")
    print("📊 Expected: MNIST ~99%, Shape ~95%+")
    
    # Create directories
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    input("\n⏸  Press ENTER to start training...")
    
    total_start = time.time()
    
    try:
        # Train MNIST
        mnist_model, mnist_acc, mnist_time = train_mnist()
        
        # Train Shape
        shape_model, shape_acc, shape_time = train_shape()
        
        # Summary
        total_time = time.time() - total_start
        
        print("\n" + "="*70)
        print("🎉 TRAINING COMPLETE!")
        print("="*70)
        
        print(f"\n⏱️  Total Time: {total_time:.1f} seconds")
        print(f"\n📊 Results:")
        print(f"   • MNIST: {mnist_acc*100:.2f}% accuracy ({mnist_time:.1f}s)")
        print(f"   • Shape: {shape_acc*100:.2f}% accuracy ({shape_time:.1f}s)")
        
        print(f"\n💾 Models saved:")
        print(f"   • {MNIST_MODEL_PATH}")
        print(f"   • {SHAPE_MODEL_PATH}")
        
        print(f"\n🧪 Test models:")
        print(f"   python test_models.py")
        
        print(f"\n🚀 Run application:")
        print(f"   python app/fast_gui.py")
        
        print("\n💡 Tips for using GUI:")
        print("   • For DIGITS: Draw LARGE and CLEAR")
        print("   • For SHAPES: Fill completely, centered")
        
        print("\n" + "="*70)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)