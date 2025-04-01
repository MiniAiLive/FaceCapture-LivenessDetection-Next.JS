import numpy as np
import cv2 as cv
import onnxruntime as ort
import os
from cryptography.fernet import Fernet

# Load encryption key from environment variable
MODEL_KEY = os.getenv("MODEL_KEY")
if not MODEL_KEY:
    raise ValueError("MODEL_KEY environment variable not set")

cipher_suite = Fernet(MODEL_KEY.encode())

# Path to encrypted model
encrypted_model_path = os.path.join(os.path.dirname(__file__), "liveness_model.onnx.enc")

# Load and decrypt the model
with open(encrypted_model_path, "rb") as f:
    encrypted_data = f.read()
decrypted_data = cipher_suite.decrypt(encrypted_data)

# Create ONNX Runtime session from decrypted bytes
ort_session = ort.InferenceSession(decrypted_data)
input_name = ort_session.get_inputs()[0].name  # 'data'
outputs = ort_session.get_outputs()

def get_liveness(img):
    """
    Predicts liveness score from input image
    Args:
        img: Input image (320x320x3) in BGR format
    Returns:
        numpy array with liveness prediction scores
    """
    # Preprocess image and convert to NCHW format
    input_blob = img.reshape(1, 3, 320, 320).astype(np.float32)
    
    # Run inference
    out = ort_session.run([outputs[0].name], input_feed={input_name: input_blob})
    
    return out[0]