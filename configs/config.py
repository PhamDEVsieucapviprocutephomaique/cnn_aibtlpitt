"""
Configuration file - Tối ưu cho tốc độ cao
"""
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
MNIST_DIR = os.path.join(DATA_DIR, 'mnist')
SHAPES_DIR = os.path.join(DATA_DIR, 'shapes')

# Model paths
MNIST_MODEL_PATH = os.path.join(MODELS_DIR, 'mnist_optimized.h5')
SHAPE_MODEL_PATH = os.path.join(MODELS_DIR, 'shape_optimized.h5')
MNIST_TFLITE_PATH = os.path.join(MODELS_DIR, 'mnist_optimized.tflite')
SHAPE_TFLITE_PATH = os.path.join(MODELS_DIR, 'shape_optimized.tflite')

# Training config
MNIST_CONFIG = {
    'img_size': (28, 28),
    'batch_size': 128,
    'epochs': 10,
    'learning_rate': 0.001,
    'num_classes': 10,
    'use_mixed_precision': True,  # Tăng tốc training
}

SHAPE_CONFIG = {
    'img_size': (64, 64),  # Nhỏ hơn để nhanh hơn
    'batch_size': 64,
    'epochs': 15,
    'learning_rate': 0.001,
    'num_classes': 3,  # circle, rectangle, triangle
    'use_mixed_precision': True,
}

# Preprocessing config
PREPROCESS_CONFIG = {
    'use_cache': True,
    'cache_size': 100,  # Cache 100 ảnh gần nhất
    'use_gpu': True,
    'num_threads': 4,
}

# UI config
UI_CONFIG = {
    'window_size': (1000, 700),
    'preview_size': (400, 400),
    'result_font_size': 24,
    'bg_color': '#2b2b2b',
    'text_color': '#ffffff',
}

# Optimization flags
OPTIMIZATION = {
    'use_tflite': False,  # Dùng TFLite cho inference nhanh
    'use_quantization': False,  # INT8 quantization
    'use_multithreading': True,
    'enable_xla': False,  # XLA compilation
}

# Shape classes
SHAPE_CLASSES = ['circle', 'rectangle', 'triangle']
MNIST_CLASSES = list(range(10))