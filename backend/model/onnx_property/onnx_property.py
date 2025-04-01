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
encrypted_model_path = os.path.join(os.path.dirname(__file__), "age_gender_model.onnx.enc")

# Load and decrypt the model
try:
    with open(encrypted_model_path, "rb") as f:
        encrypted_data = f.read()
    decrypted_data = cipher_suite.decrypt(encrypted_data)
    ort_session = ort.InferenceSession(decrypted_data)
except Exception as e:
    raise RuntimeError(f"Failed to load age/gender model: {str(e)}")

input_name = ort_session.get_inputs()[0].name  # 'data'
outputs = ort_session.get_outputs()

# Age reference tables
AGE_REFERENCE = [
    [0,     3.0,    10.5,   20.5,   30.5,   40.5,   53.0,   70.0],  # Age bins
    [5.0,   15.0,   25.0,   35.0,   45.0,   60.0,   80.0,   0],     # Upper bounds
    [0.0,   6.0,    16.0,   26.0,   36.0,   46.0,   61.0,   0],     # Lower bounds
]

def get_age_gender(output):
    """
    Calculate age and gender from model output
    Args:
        output: Model output containing age and gender predictions
    Returns:
        tuple: (age, gender) where age is in years and gender is prediction score
    """
    age_level = output[0]
    age_matrix = output[4]
    gender = output[2]
    
    # Calculate age value based on age level
    if age_level > 0:
        if age_level == 6:
            age_val = age_matrix[6] * 70.0 + age_matrix[5] * 53.0
        else:
            age_val = (age_matrix[age_level] * AGE_REFERENCE[0][age_level+1] + 
                      age_matrix[age_level - 1] * AGE_REFERENCE[0][age_level] + 
                      age_matrix[age_level + 1] * AGE_REFERENCE[0][age_level + 2])
    else:
        age_val = age_matrix[1] * 10.5 + age_matrix[0] * 0.0

    # Apply bounds checking
    if age_val > AGE_REFERENCE[1][age_level]:
        age_val = AGE_REFERENCE[1][age_level]
    if age_val < AGE_REFERENCE[2][age_level]:
        age_val = AGE_REFERENCE[2][age_level]
        
    return age_val, gender

def get_property(img):
    """
    Predict age and gender properties from input image
    Args:
        img: Input image (60x60x3) in BGR format
    Returns:
        Model outputs containing age and gender predictions
    """
    # Preprocess image and convert to NCHW format
    input_blob = img.reshape(1, 3, 60, 60).astype(np.float32)
    
    # Run inference - get all outputs
    out = ort_session.run(
        [outputs[0].name, outputs[1].name, outputs[2].name, 
         outputs[3].name, outputs[4].name],
        input_feed={input_name: input_blob}
    )
    return out