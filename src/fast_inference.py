"""
FAST Inference Engine - Prediction nhanh
"""
import os
import sys
import numpy as np
from tensorflow import keras

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import *
from src.fast_preprocessor import get_fast_preprocessor


class FastInferenceEngine:
    """Fast inference với 48x48 shapes"""
    
    def __init__(self):
        self.mnist_model = None
        self.shape_model = None
        self.preprocessor = get_fast_preprocessor()
        self.load_models()
    
    def load_models(self):
        """Load models"""
        print("Loading models...")
        
        if os.path.exists(MNIST_MODEL_PATH):
            self.mnist_model = keras.models.load_model(MNIST_MODEL_PATH)
            print("✓ MNIST model loaded")
        else:
            print("⚠ MNIST model not found!")
        
        if os.path.exists(SHAPE_MODEL_PATH):
            self.shape_model = keras.models.load_model(SHAPE_MODEL_PATH)
            print("✓ Shape model loaded (48x48)")
        else:
            print("⚠ Shape model not found!")
    
    def predict_mnist(self, image):
        """Predict digit"""
        if self.mnist_model is None:
            return None, 0.0
        
        processed = self.preprocessor.preprocess_for_mnist(image)
        predictions = self.mnist_model.predict(processed, verbose=0)[0]
        
        digit = int(np.argmax(predictions))
        confidence = float(predictions[digit])
        
        return digit, confidence
    
    def predict_shape(self, image):
        """Predict shape"""
        if self.shape_model is None:
            return None, 0.0
        
        processed = self.preprocessor.preprocess_for_shape(image)
        predictions = self.shape_model.predict(processed, verbose=0)[0]
        
        shape_idx = int(np.argmax(predictions))
        shape = SHAPE_CLASSES[shape_idx]
        confidence = float(predictions[shape_idx])
        
        return shape, confidence
    
    def predict_auto(self, image):
        """Auto predict - IMPROVED logic"""
        # Detect type
        detected_type = self.preprocessor.smart_detect_type(image)
        
        print(f"🔍 Detected: {detected_type}")
        
        # Try BOTH models and choose best
        digit, digit_conf = self.predict_mnist(image)
        shape, shape_conf = self.predict_shape(image)
        
        print(f"   MNIST: {digit} ({digit_conf:.2%})")
        print(f"   Shape: {shape} ({shape_conf:.2%})")
        
        # Decision logic
        if detected_type == 'mnist':
            # Prefer digit, but check if shape is much more confident
            if shape_conf > digit_conf + 0.3 and shape_conf > 0.8:
                return {
                    'type': 'shape',
                    'result': shape,
                    'confidence': shape_conf,
                    'raw_value': shape
                }
            else:
                return {
                    'type': 'digit',
                    'result': str(digit),
                    'confidence': digit_conf,
                    'raw_value': digit
                }
        else:
            # Prefer shape, but check if it's actually digit 0
            if shape == 'circle' and digit == 0:
                # Circle vs 0 confusion - check confidence
                if digit_conf > shape_conf:
                    return {
                        'type': 'digit',
                        'result': '0',
                        'confidence': digit_conf,
                        'raw_value': 0
                    }
            
            # If shape confidence is high, return shape
            if shape_conf > 0.7:
                return {
                    'type': 'shape',
                    'result': shape,
                    'confidence': shape_conf,
                    'raw_value': shape
                }
            # Otherwise, if digit confidence is decent, return digit
            elif digit_conf > 0.5:
                return {
                    'type': 'digit',
                    'result': str(digit),
                    'confidence': digit_conf,
                    'raw_value': digit
                }
            else:
                # Return the more confident one
                if shape_conf > digit_conf:
                    return {
                        'type': 'shape',
                        'result': shape,
                        'confidence': shape_conf,
                        'raw_value': shape
                    }
                else:
                    return {
                        'type': 'digit',
                        'result': str(digit),
                        'confidence': digit_conf,
                        'raw_value': digit
                    }


_engine = None

def get_fast_engine():
    """Singleton"""
    global _engine
    if _engine is None:
        _engine = FastInferenceEngine()
    return _engine