"""
Ultra Inference Engine - Prediction với ensemble và confidence scoring
"""
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import *
from src.preprocessor import get_preprocessor


class ShapeFeatureExtractor:
    """Trích xuất đặc trưng hình học để hỗ trợ prediction"""
    
    @staticmethod
    def extract_features(img):
        """Extract geometric features"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img.copy()
        
        # Threshold
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Largest contour
        largest = max(contours, key=cv2.contourArea)
        
        # Features
        area = cv2.contourArea(largest)
        perimeter = cv2.arcLength(largest, True)
        
        if perimeter == 0:
            return None
        
        # Circularity
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Convexity
        hull = cv2.convexHull(largest)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Vertices
        epsilon = 0.04 * perimeter
        approx = cv2.approxPolyDP(largest, epsilon, True)
        num_vertices = len(approx)
        
        # Aspect ratio
        x, y, w, h = cv2.boundingRect(largest)
        aspect_ratio = w / h if h > 0 else 1
        
        # Extent
        rect_area = w * h
        extent = area / rect_area if rect_area > 0 else 0
        
        return {
            'circularity': circularity,
            'solidity': solidity,
            'num_vertices': num_vertices,
            'aspect_ratio': aspect_ratio,
            'extent': extent,
            'area': area
        }
    
    @staticmethod
    def analyze_shape(features):
        """Analyze shape từ features"""
        if features is None:
            return None, 0.0
        
        scores = {'circle': 0, 'square': 0, 'triangle': 0}
        
        # Circle scoring
        if features['circularity'] > 0.85:
            scores['circle'] += 3
        elif features['circularity'] > 0.75:
            scores['circle'] += 2
        elif features['circularity'] > 0.65:
            scores['circle'] += 1
        
        if features['solidity'] > 0.95:
            scores['circle'] += 2
        
        # Square scoring
        if features['num_vertices'] == 4:
            scores['square'] += 3
        elif features['num_vertices'] in [3, 5]:
            scores['square'] += 1
        
        if 0.8 < features['aspect_ratio'] < 1.2:
            scores['square'] += 2
        
        if features['extent'] > 0.8:
            scores['square'] += 2
        
        # Triangle scoring
        if features['num_vertices'] == 3:
            scores['triangle'] += 4
        elif features['num_vertices'] == 4:
            scores['triangle'] += 1
        
        if 0.6 < features['solidity'] < 0.9:
            scores['triangle'] += 2
        
        # Get best
        best_shape = max(scores, key=scores.get)
        confidence = scores[best_shape] / 10.0  # Normalize
        
        return best_shape, min(confidence, 1.0)


class UltraInferenceEngine:
    """Inference engine với ensemble và feature analysis"""
    
    def __init__(self):
        self.mnist_model = None
        self.shape_model = None
        self.preprocessor = get_preprocessor()
        self.feature_extractor = ShapeFeatureExtractor()
        self.load_models()
    
    def load_models(self):
        """Load trained models"""
        print("Loading ULTRA models...")
        
        if os.path.exists(MNIST_MODEL_PATH):
            self.mnist_model = keras.models.load_model(MNIST_MODEL_PATH)
            print("✓ MNIST model loaded")
        else:
            print("⚠ MNIST model not found!")
        
        if os.path.exists(SHAPE_MODEL_PATH):
            self.shape_model = keras.models.load_model(SHAPE_MODEL_PATH)
            print("✓ Shape model loaded")
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
    
    def predict_shape_with_ensemble(self, image):
        """Predict shape với ensemble (model + features)"""
        if self.shape_model is None:
            return None, 0.0
        
        # Model prediction
        processed = self.preprocessor.preprocess_for_shape(image)
        model_preds = self.shape_model.predict(processed, verbose=0)[0]
        
        model_shape_idx = int(np.argmax(model_preds))
        model_shape = SHAPE_CLASSES[model_shape_idx]
        model_conf = float(model_preds[model_shape_idx])
        
        # Feature-based prediction
        features = self.feature_extractor.extract_features(image)
        feature_shape, feature_conf = self.feature_extractor.analyze_shape(features)
        
        # Ensemble decision
        if feature_shape is None:
            return model_shape, model_conf
        
        # Nếu cả hai agree
        if model_shape == feature_shape:
            # Boost confidence
            final_conf = min(1.0, model_conf * 0.7 + feature_conf * 0.3 + 0.1)
            return model_shape, final_conf
        
        # Nếu disagree, trust model hơn nhưng giảm confidence
        if model_conf > 0.7:
            return model_shape, model_conf * 0.8
        else:
            # Model không chắc chắn, xem feature
            if feature_conf > 0.5:
                # Weighted vote
                if model_conf > feature_conf:
                    return model_shape, model_conf * 0.7
                else:
                    return feature_shape, feature_conf * 0.7
            else:
                return model_shape, model_conf * 0.6
    
    def smart_type_detection(self, image):
        """Smart detection loại input"""
        if isinstance(image, np.ndarray):
            img = image.copy()
        else:
            img = np.array(image)
        
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        # Analyze
        features = self.feature_extractor.extract_features(gray)
        
        if features is None:
            return 'mnist'
        
        # Heuristics
        # Nếu circularity cao -> có thể là circle hoặc số 0
        if features['circularity'] > 0.75:
            # Nếu area nhỏ và có nhiều holes -> digit
            if features['area'] < 5000 and features['solidity'] < 0.85:
                return 'mnist'
            else:
                return 'shape'
        
        # Nếu có 3-4 vertices rõ ràng -> shape
        if features['num_vertices'] in [3, 4] and features['solidity'] > 0.75:
            return 'shape'
        
        # Nếu shape phức tạp -> digit
        if features['num_vertices'] > 5 or features['solidity'] < 0.7:
            return 'mnist'
        
        # Default
        return 'shape' if features['area'] > 8000 else 'mnist'
    
    def predict_auto(self, image):
        """Auto predict với intelligent type detection"""
        # Detect type
        detected_type = self.smart_type_detection(image)
        
        print(f"🔍 Detected type: {detected_type}")
        
        if detected_type == 'mnist':
            digit, conf = self.predict_mnist(image)
            
            # Nếu confidence thấp và digit là 0, thử shape
            if digit == 0 and conf < 0.7:
                shape, shape_conf = self.predict_shape_with_ensemble(image)
                if shape == 'circle' and shape_conf > conf:
                    return {
                        'type': 'shape',
                        'result': 'circle',
                        'confidence': shape_conf,
                        'raw_value': 'circle'
                    }
            
            return {
                'type': 'digit',
                'result': str(digit),
                'confidence': conf,
                'raw_value': digit
            }
        else:
            shape, conf = self.predict_shape_with_ensemble(image)
            
            # Nếu shape là circle với confidence thấp, thử digit 0
            if shape == 'circle' and conf < 0.6:
                digit, digit_conf = self.predict_mnist(image)
                if digit == 0 and digit_conf > conf:
                    return {
                        'type': 'digit',
                        'result': '0',
                        'confidence': digit_conf,
                        'raw_value': 0
                    }
            
            return {
                'type': 'shape',
                'result': shape,
                'confidence': conf,
                'raw_value': shape
            }


_engine = None

def get_engine():
    """Get singleton engine"""
    global _engine
    if _engine is None:
        _engine = UltraInferenceEngine()
    return _engine


if __name__ == "__main__":
    print("Ultra inference engine loaded!")