"""
Configuration - Updated for ULTRA training
"""
import os

# BASE PATHS
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# MODEL PATHS
MNIST_MODEL_PATH = os.path.join(MODELS_DIR, 'mnist_model.h5')
MNIST_TFLITE_PATH = os.path.join(MODELS_DIR, 'mnist_model.tflite')
SHAPE_MODEL_PATH = os.path.join(MODELS_DIR, 'shape_model.h5')
SHAPE_TFLITE_PATH = os.path.join(MODELS_DIR, 'shape_model.tflite')

# MNIST CONFIG
MNIST_CONFIG = {
    'img_size': (28, 28),
    'batch_size': 128,
    'epochs': 50,
    'learning_rate': 0.0003,
    'num_classes': 10
}

# SHAPE CONFIG - Optimized for maximum accuracy
SHAPE_CONFIG = {
    'img_size': (64, 64),
    'batch_size': 32,
    'epochs': 150,
    'learning_rate': 0.0001,
    'num_classes': 3
}

# SHAPE CLASSES
SHAPE_CLASSES = ['circle', 'square', 'triangle']

# PREPROCESSING CONFIG
PREPROCESS_CONFIG = {
    'use_cache': True,
    'cache_size': 100,
    'target_size_mnist': (28, 28),
    'target_size_shape': (64, 64),
    'use_advanced_preprocessing': True,
    'use_clahe': True,
    'use_bilateral_filter': True,
    'use_morphological_ops': True
}

# OPTIMIZATION CONFIG
OPTIMIZATION = {
    'use_tflite': False,  # Disable for maximum accuracy
    'use_quantization': False,
    'use_gpu': True,
    'num_threads': 4
}

# TRAINING CONFIG
TRAINING_CONFIG = {
    'use_data_augmentation': True,
    'samples_per_class': 8000,
    'validation_split': 0.15,
    'early_stopping_patience': 25,
    'reduce_lr_patience': 10
}

# Create directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(MODELS_DIR, 'checkpoints'), exist_ok=True)