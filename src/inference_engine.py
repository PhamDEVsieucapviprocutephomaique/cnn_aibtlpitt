"""
Inference Engine - Prediction siêu nhanh với multi-threading và caching
"""
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import threading
import queue
from collections import OrderedDict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import *
from src.preprocessor import get_preprocessor


class PredictionCache:
    """Cache predictions với LRU"""
    
    def __init__(self, max_size=50):
        self.cache = OrderedDict()
        self.max_size = max_size
    
    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)


class FastInferenceEngine:
    """Engine cho inference siêu nhanh"""
    
    def __init__(self):
        self.mnist_model = None
        self.shape_model = None
        self.mnist_interpreter = None
        self.shape_interpreter = None
        self.preprocessor = get_preprocessor()
        self.prediction_cache = PredictionCache()
        self.load_models()
    
    def load_models(self):
        """Load models - ưu tiên TFLite"""
        print("Loading models...")
        
        if OPTIMIZATION['use_tflite'] and os.path.exists(MNIST_TFLITE_PATH):
            self.mnist_interpreter = tf.lite.Interpreter(model_path=MNIST_TFLITE_PATH)
            self.mnist_interpreter.allocate_tensors()
            print("MNIST TFLite model loaded")
        elif os.path.exists(MNIST_MODEL_PATH):
            self.mnist_model = keras.models.load_model(MNIST_MODEL_PATH)
            print("MNIST Keras model loaded")
        else:
            print("WARNING: MNIST model not found!")
        
        if OPTIMIZATION['use_tflite'] and os.path.exists(SHAPE_TFLITE_PATH):
            self.shape_interpreter = tf.lite.Interpreter(model_path=SHAPE_TFLITE_PATH)
            self.shape_interpreter.allocate_tensors()
            print("Shape TFLite model loaded")
        elif os.path.exists(SHAPE_MODEL_PATH):
            self.shape_model = keras.models.load_model(SHAPE_MODEL_PATH)
            print("Shape Keras model loaded")
        else:
            print("WARNING: Shape model not found!")
    
    def predict_with_tflite(self, interpreter, input_data):
        """Predict sử dụng TFLite interpreter"""
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])
        return output_data
    
    def predict_mnist(self, image, use_cache=True):
        """
        Predict chữ số - siêu nhanh
        Returns: (digit, confidence)
        """
        cache_key = hash(image.tobytes()) if use_cache else None
        
        if use_cache and cache_key:
            cached = self.prediction_cache.get(f"mnist_{cache_key}")
            if cached:
                return cached
        
        processed = self.preprocessor.preprocess_for_mnist(image)
        
        if self.mnist_interpreter:
            predictions = self.predict_with_tflite(
                self.mnist_interpreter, 
                processed.astype(np.float32)
            )[0]
        elif self.mnist_model:
            predictions = self.mnist_model.predict(processed, verbose=0)[0]
        else:
            return None, 0.0
        
        digit = int(np.argmax(predictions))
        confidence = float(predictions[digit])
        
        result = (digit, confidence)
        
        if use_cache and cache_key:
            self.prediction_cache.put(f"mnist_{cache_key}", result)
        
        return result
    
    def predict_shape(self, image, use_cache=True):
        """
        Predict hình dạng - siêu nhanh
        Returns: (shape_name, confidence)
        """
        cache_key = hash(image.tobytes()) if use_cache else None
        
        if use_cache and cache_key:
            cached = self.prediction_cache.get(f"shape_{cache_key}")
            if cached:
                return cached
        
        processed = self.preprocessor.preprocess_for_shape(image)
        
        if self.shape_interpreter:
            predictions = self.predict_with_tflite(
                self.shape_interpreter,
                processed.astype(np.float32)
            )[0]
        elif self.shape_model:
            predictions = self.shape_model.predict(processed, verbose=0)[0]
        else:
            return None, 0.0
        
        shape_idx = int(np.argmax(predictions))
        shape_name = SHAPE_CLASSES[shape_idx]
        confidence = float(predictions[shape_idx])
        
        result = (shape_name, confidence)
        
        if use_cache and cache_key:
            self.prediction_cache.put(f"shape_{cache_key}", result)
        
        return result
    
    def predict_auto(self, image):
        """
        Auto detect và predict - siêu nhanh
        Returns: dict với type, result, confidence
        """
        processed, detected_type = self.preprocessor.auto_detect_and_preprocess(image)
        
        if detected_type == 'mnist':
            digit, conf = self.predict_mnist(image)
            return {
                'type': 'digit',
                'result': str(digit),
                'confidence': conf,
                'raw_value': digit
            }
        else:
            shape, conf = self.predict_shape(image)
            return {
                'type': 'shape',
                'result': shape,
                'confidence': conf,
                'raw_value': shape
            }


class AsyncInferenceEngine:
    """Async inference với threading cho UI non-blocking"""
    
    def __init__(self):
        self.engine = FastInferenceEngine()
        self.request_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.worker_thread = None
        self.running = False
    
    def start(self):
        """Start worker thread"""
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
    
    def stop(self):
        """Stop worker thread"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=1)
    
    def _worker(self):
        """Worker thread cho predictions"""
        while self.running:
            try:
                request = self.request_queue.get(timeout=0.1)
                image, request_id = request
                
                result = self.engine.predict_auto(image)
                
                self.result_queue.put((request_id, result))
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Prediction error: {e}")
    
    def predict_async(self, image, request_id=None):
        """Submit async prediction request"""
        if request_id is None:
            import time
            request_id = int(time.time() * 1000)
        
        self.request_queue.put((image, request_id))
        return request_id
    
    def get_result(self, timeout=None):
        """Get prediction result"""
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None


_engine = None
_async_engine = None

def get_engine():
    """Get singleton engine"""
    global _engine
    if _engine is None:
        _engine = FastInferenceEngine()
    return _engine

def get_async_engine():
    """Get singleton async engine"""
    global _async_engine
    if _async_engine is None:
        _async_engine = AsyncInferenceEngine()
        _async_engine.start()
    return _async_engine


if __name__ == "__main__":
    import time
    
    print("Testing inference engine...")
    engine = FastInferenceEngine()
    
    test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    for _ in range(5):
        engine.predict_auto(test_img)
    
    times = []
    for _ in range(50):
        start = time.perf_counter()
        result = engine.predict_auto(test_img)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    
    print(f"Average inference time: {np.mean(times):.2f}ms")
    print(f"Min: {np.min(times):.2f}ms, Max: {np.max(times):.2f}ms")
    print("\nInference engine test successful!")