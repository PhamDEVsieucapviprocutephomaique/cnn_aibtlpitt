"""
ULTRA TRAINING - Script chính để train toàn bộ hệ thống
Chạy file này để train models với độ chính xác CỰC CAO
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import training modules
from src.super_trainer import train_ultra_mnist, train_ultra_shape
from configs.config import MODELS_DIR


def main():
    """Main training pipeline"""
    print("=" * 80)
    print(" " * 20 + "🎯 ULTRA NEURAL RECOGNITION")
    print(" " * 15 + "MAXIMUM ACCURACY TRAINING SYSTEM")
    print("=" * 80)
    
    print("\n📋 Training Configuration:")
    print("   • MNIST: ResNet-like architecture with residual blocks")
    print("   • Shape: Attention-based CNN with 8000 samples/class")
    print("   • Advanced preprocessing: CLAHE, bilateral filter, edge detection")
    print("   • Ensemble: Model + geometric feature analysis")
    print("   • Data augmentation: Heavy augmentation for robustness")
    
    input("\n⏸  Press ENTER to start training (this will take time)...")
    
    # Create directories
    print("\n📁 Creating directories...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(os.path.join(MODELS_DIR, 'checkpoints'), exist_ok=True)
    print("✓ Directories ready")
    
    try:
        # Train MNIST
        print("\n" + "="*80)
        print("STEP 1/2: Training MNIST Model")
        print("="*80)
        mnist_model, mnist_history = train_ultra_mnist()
        print("\n✅ MNIST training complete!")
        
        # Train Shape
        print("\n" + "="*80)
        print("STEP 2/2: Training Shape Model")
        print("="*80)
        shape_model, shape_history = train_ultra_shape()
        print("\n✅ Shape training complete!")
        
        # Summary
        print("\n" + "="*80)
        print(" " * 25 + "🎉 TRAINING COMPLETE!")
        print("="*80)
        
        print("\n📊 Training Summary:")
        print(f"   • MNIST Model: {MODELS_DIR}/mnist_model.h5")
        print(f"   • Shape Model: {MODELS_DIR}/shape_model.h5")
        print("   • Both models trained with ULTRA accuracy techniques")
        
        print("\n🚀 Next Steps:")
        print("   1. Run the application:")
        print("      python app/super_gui.py")
        print("\n   2. Draw shapes or digits and test accuracy")
        print("   3. The system will automatically choose the best model")
        
        print("\n💡 Tips for best results:")
        print("   • Draw shapes large and centered")
        print("   • Make circles as round as possible")
        print("   • Draw triangles with clear 3 sides")
        print("   • Draw squares with clear 4 corners")
        print("   • For digits, draw them clearly")
        
        print("\n" + "="*80)
        
    except Exception as e:
        print(f"\n❌ ERROR during training: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)