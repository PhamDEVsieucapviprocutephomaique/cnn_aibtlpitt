"""
Training Script - Updated imports
Chạy file này để train models lần đầu
"""
import os
import sys

# Setup paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_all_data
from src.model_builder import train_mnist_model, train_shape_model
from configs.config import MODELS_DIR


def main():
    """Main training pipeline"""
    print("=" * 60)
    print("NEURAL RECOGNITION - TRAINING PIPELINE")
    print("=" * 60)
    
    # Create directories
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(os.path.join(MODELS_DIR, 'checkpoints'), exist_ok=True)
    
    # Step 1: Load data
    print("\n[1/3] Loading and generating datasets...")
    mnist_data, shape_data = load_all_data()
    print("✓ Data loaded successfully")
    
    # Step 2: Train MNIST
    print("\n[2/3] Training MNIST model...")
    mnist_model, mnist_history = train_mnist_model(mnist_data)
    print("✓ MNIST model trained successfully")
    
    # Step 3: Train Shape
    print("\n[3/3] Training Shape detection model...")
    shape_model, shape_history = train_shape_model(shape_data)
    print("✓ Shape model trained successfully")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print("\nModels saved to:", MODELS_DIR)
    print("\nYou can now run the application:")
    print("  python app/enhanced_gui.py")
    print("=" * 60)


if __name__ == "__main__":
    main()