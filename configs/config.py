"""
Configuration - Updated for better shape recognition
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
    'batch_size': 32,
    'epochs': 10,
    'learning_rate': 0.001,
    'num_classes': 10
}

# SHAPE CONFIG - Updated for better accuracy
SHAPE_CONFIG = {
    'img_size': (64, 64),
    'batch_size': 32,
    'epochs': 20,  # Increased epochs
    'learning_rate': 0.0005,  # Lower learning rate
    'num_classes': 3
}

# SHAPE CLASSES - More specific
SHAPE_CLASSES = ['circle', 'square', 'triangle']

# PREPROCESSING CONFIG
PREPROCESS_CONFIG = {
    'use_cache': True,
    'cache_size': 100,
    'target_size_mnist': (28, 28),
    'target_size_shape': (64, 64)
}

# OPTIMIZATION CONFIG
OPTIMIZATION = {
    'use_tflite': True,
    'use_quantization': True,
    'use_gpu': True,
    'num_threads': 4
}

# Create directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)