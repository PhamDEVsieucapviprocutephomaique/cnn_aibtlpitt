"""
FAST Configuration - Tối ưu cho tốc độ training
"""
import os

# BASE PATHS
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# MODEL PATHS
MNIST_MODEL_PATH = os.path.join(MODELS_DIR, 'mnist_model.h5')
SHAPE_MODEL_PATH = os.path.join(MODELS_DIR, 'shape_model.h5')

# FAST MNIST CONFIG
MNIST_CONFIG = {
    'img_size': (28, 28),
    'batch_size': 256,      # Batch lớn = nhanh
    'epochs': 10,           # Ít epochs
    'learning_rate': 0.001, # LR cao = train nhanh
    'num_classes': 10,
    'samples': 30000        # Subsample
}

# FAST SHAPE CONFIG - 48x48 thay vì 64x64
SHAPE_CONFIG = {
    'img_size': (48, 48),   # Giảm kích thước
    'batch_size': 64,
    'epochs': 30,           # Đủ cho 95%+
    'learning_rate': 0.001,
    'num_classes': 3,
    'samples_per_class': 1500  # Giảm từ 8000
}

# SHAPE CLASSES
SHAPE_CLASSES = ['circle', 'square', 'triangle']

# FAST PREPROCESSING
PREPROCESS_CONFIG = {
    'use_cache': False,              # Disable cache
    'target_size_mnist': (28, 28),
    'target_size_shape': (48, 48),   # 48x48
    'use_advanced_preprocessing': False,  # Simple preprocessing
    'use_clahe': False,
    'use_bilateral_filter': False,
    'use_morphological_ops': False
}

# NO OPTIMIZATION (để đơn giản)
OPTIMIZATION = {
    'use_tflite': False,
    'use_quantization': False,
    'use_gpu': True,
    'num_threads': 4
}

# FAST TRAINING
TRAINING_CONFIG = {
    'use_data_augmentation': True,   # Simple augmentation
    'samples_per_class': 1500,
    'validation_split': 0.2,
    'early_stopping_patience': 5,    # Ít patience
    'reduce_lr_patience': 3
}

# Create directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)