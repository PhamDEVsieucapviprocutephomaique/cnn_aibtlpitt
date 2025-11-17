"""
Model Builder - Complete version với tất cả functions
"""
import os
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import *


# Enable XLA compilation cho tốc độ cao hơn
tf.config.optimizer.set_jit(True)


class OptimizedMNISTModel:
    """MNIST Model tối ưu - siêu nhanh"""
    
    @staticmethod
    def build():
        """Build lightweight CNN cho MNIST"""
        model = models.Sequential([
            # Conv block 1 - ít filters hơn
            layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            
            # Conv block 2
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            
            # Dense layers - compact hơn
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(10, activation='softmax')
        ])
        
        # Compile với optimizer tối ưu
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=MNIST_CONFIG['learning_rate']),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model


class EnhancedShapeModel:
    """Enhanced Shape Model với architecture tốt hơn"""
    
    @staticmethod
    def build():
        """Build better CNN cho shapes"""
        img_size = SHAPE_CONFIG['img_size'][0]
        
        model = models.Sequential([
            # Conv block 1
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 1)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Conv block 2
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Conv block 3
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(3, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=SHAPE_CONFIG['learning_rate']),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model


class ModelOptimizer:
    """Tối ưu models sau training"""
    
    @staticmethod
    def convert_to_tflite(model, save_path, quantize=True):
        """Convert sang TFLite với INT8 quantization"""
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        
        tflite_model = converter.convert()
        
        with open(save_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"TFLite model saved: {save_path}")
        return tflite_model
    
    @staticmethod
    def benchmark_model(model, test_data, num_runs=100):
        """Benchmark inference speed"""
        import time
        
        x_test, _ = test_data
        sample = x_test[:1]
        
        # Warmup
        for _ in range(10):
            model.predict(sample, verbose=0)
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            model.predict(sample, verbose=0)
            times.append(time.perf_counter() - start)
        
        avg_time = np.mean(times) * 1000  # ms
        print(f"Average inference time: {avg_time:.2f}ms")
        return avg_time


def train_mnist_model(data, save_path=MNIST_MODEL_PATH):
    """Train MNIST model với tối ưu"""
    (x_train, y_train), (x_test, y_test) = data
    
    print("\n=== Training MNIST Model ===")
    model = OptimizedMNISTModel.build()
    print(model.summary())
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2
        )
    ]
    
    # Training
    history = model.fit(
        x_train, y_train,
        batch_size=MNIST_CONFIG['batch_size'],
        epochs=MNIST_CONFIG['epochs'],
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_acc*100:.2f}%")
    
    # Save
    model.save(save_path)
    print(f"Model saved: {save_path}")
    
    # Convert to TFLite
    if OPTIMIZATION['use_tflite']:
        tflite_path = MNIST_TFLITE_PATH
        ModelOptimizer.convert_to_tflite(model, tflite_path, OPTIMIZATION['use_quantization'])
    
    # Benchmark
    ModelOptimizer.benchmark_model(model, (x_test, y_test))
    
    return model, history


def train_shape_model(data, save_path=SHAPE_MODEL_PATH):
    """Train enhanced Shape model"""
    (x_train, y_train), (x_test, y_test) = data
    
    print("\n=== Training Enhanced Shape Model ===")
    model = EnhancedShapeModel.build()
    print(model.summary())
    
    # Enhanced callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,  # Increased patience
            restore_best_weights=True,
            min_delta=0.001
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7
        ),
        keras.callbacks.ModelCheckpoint(
            os.path.join(MODELS_DIR, 'checkpoints', 'shape_best.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Training với data augmentation
    history = model.fit(
        x_train, y_train,
        batch_size=SHAPE_CONFIG['batch_size'],
        epochs=SHAPE_CONFIG['epochs'],
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1,
        shuffle=True
    )
    
    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_acc*100:.2f}%")
    
    # Save
    model.save(save_path)
    print(f"Model saved: {save_path}")
    
    # Convert to TFLite
    if OPTIMIZATION['use_tflite']:
        tflite_path = SHAPE_TFLITE_PATH
        ModelOptimizer.convert_to_tflite(model, tflite_path, OPTIMIZATION['use_quantization'])
    
    # Benchmark
    ModelOptimizer.benchmark_model(model, (x_test, y_test))
    
    return model, history


if __name__ == "__main__":
    from data_loader import load_all_data
    
    print("Loading data...")
    mnist_data, shape_data = load_all_data()
    
    print("\nTraining models...")
    mnist_model, _ = train_mnist_model(mnist_data)
    shape_model, _ = train_shape_model(shape_data)
    
    print("\n=== Training Complete ===")