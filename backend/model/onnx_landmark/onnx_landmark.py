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
encrypted_model_path = os.path.join(os.path.dirname(__file__), "landmark_model.onnx.enc")

# Load and decrypt the model
with open(encrypted_model_path, "rb") as f:
    encrypted_data = f.read()
decrypted_data = cipher_suite.decrypt(encrypted_data)

# Create ONNX Runtime session from decrypted bytes
ort_session = ort.InferenceSession(decrypted_data)
input_name = ort_session.get_inputs()[0].name  # 'data'
outputs = ort_session.get_outputs()

coeff = [
    0.237826, 0.483403, 0.23452599, 0.583606, 0.233119,
    0.68341398, 0.234304, 0.78207201, 0.24113999, 0.88146299,
    0.25445601, 0.98053998, 0.274923, 1.07964, 0.30010599,
    1.17435, 0.33293301, 1.26717, 0.37198099, 1.35163, 0.42019001,
    1.4320101, 0.47444299, 1.50337, 0.53546101, 1.57005,
    0.60001898, 1.62968, 0.669393, 1.68459, 0.743249, 1.73281,
    0.824705, 1.77175, 0.91355598, 1.79583, 1.00761, 1.80226,
    1.10069, 1.79591, 1.19055, 1.77174, 1.27108, 1.7331899,
    1.34586, 1.68451, 1.4143699, 1.6302, 1.47979, 1.56998,
    1.53985, 1.50394, 1.59488, 1.4319299, 1.64231, 1.35224,
    1.68211, 1.2671, 1.71429, 1.17494, 1.73997, 1.07954,
    1.75981, 0.98109603, 1.77353, 0.88134402, 1.77997, 0.78266799,
    1.7815, 0.68330598, 1.7797101, 0.58351398, 1.77736,
    0.48331499, 0.42815101, 0.331536, 0.49587101, 0.252868,
    0.58939898, 0.22267701, 0.69432801, 0.224355, 0.793724,
    0.248253, 0.86448902, 0.320492, 0.77540201, 0.324074,
    0.68621403, 0.31353301, 0.59634799, 0.310624, 0.51176602,
    0.32030201, 1.15092, 0.320481, 1.22167, 0.24822401,
    1.32107, 0.22433101, 1.42601, 0.222638, 1.51961, 0.25281,
    1.58725, 0.331478, 1.5037, 0.320252, 1.4190201, 0.31059,
    1.32916, 0.31349, 1.23999, 0.32404, 0.53237897, 0.51341498,
    0.57020998, 0.48187599, 0.61657101, 0.462832, 0.671148,
    0.455275, 0.72714299, 0.46329501, 0.77389199, 0.486534,
    0.80923003, 0.525123, 0.76439798, 0.53624398, 0.71833497,
    0.54415202, 0.668935, 0.54775602, 0.61870098, 0.543966,
    0.57374799, 0.53281403, 1.20622, 0.52510399, 1.24149,
    0.486514, 1.28828, 0.46329001, 1.34429, 0.45524499,
    1.39888, 0.46279699, 1.44521, 0.48181501, 1.48307, 0.51337999,
    1.44169, 0.53272998, 1.39685, 0.543935, 1.3465199, 0.547732,
    1.29711, 0.54411399, 1.25105, 0.53621697, 0.91036898,
    0.494387, 0.90331, 0.64266998, 0.876818, 0.79386699,
    0.79013002, 0.88744998, 0.82932401, 0.99425799, 0.96432298,
    1.00447, 1.05118, 1.00447, 1.1862, 0.99426401, 1.2254,
    0.88742501, 1.13869, 0.79384798, 1.11215, 0.64265603,
    1.1051199, 0.49437299, 0.74025601, 1.27909, 0.82153898,
    1.21007, 0.92972302, 1.16291, 1.0077699, 1.17785, 1.08587,
    1.16295, 1.194, 1.21008, 1.2753, 1.27907, 1.20895, 1.35777,
    1.12245, 1.413, 1.00782, 1.43376, 0.89314598, 1.41302,
    0.80662, 1.35773, 0.76710898, 1.28056, 0.881082, 1.25333,
    1.0077699, 1.25087, 1.13446, 1.2532901, 1.2483701, 1.28056,
    1.1353, 1.30952, 1.00776, 1.32513, 0.88025498, 1.30953,
]

def get_angle(Dst, faceRect):
    faceRect2 = [int(faceRect[0] - faceRect[2]*0.1), 
                int(faceRect[1] - faceRect[3]*0.1), 
                int(faceRect[2]*1.2), 
                int(faceRect[3]*1.2)]
    faceRectWithRatio = [0, 0, 0, 0, 0, 0]
    faceRectWithRatio[0] = faceRect2[2] / 64.0
    faceRectWithRatio[4] = faceRect2[3] / 64.0
    faceRectWithRatio[2] = faceRect2[0]
    faceRectWithRatio[5] = faceRect2[1]
    
    val = np.zeros((244,))
    min_max = [10000000.0, -10000000.0, 10000000.0, -10000000.0]
    
    for i in range(113):
        pos0 = (coeff[2 * i] + Dst[2 * i]) * 32.0
        pos1 = (coeff[2 * i + 1] + Dst[2 * i + 1]) * 32.0
        val[2*i] = (faceRectWithRatio[1] * pos1) + (pos0 * faceRectWithRatio[0]) + faceRectWithRatio[2]
        val[2*i+1] = (faceRectWithRatio[4] * pos1) + (pos0 * faceRectWithRatio[3]) + faceRectWithRatio[5]
        
        if min_max[0] > val[2*i]:
            min_max[0] = val[2*i]
        if min_max[1] < val[2*i]:
            min_max[1] = val[2*i]
        if min_max[2] > val[2*i+1]:
            min_max[2] = val[2*i+1]
        if min_max[3] < val[2*i+1]:
            min_max[3] = val[2*i+1]
    
    dist = min_max[1] - min_max[0]
    if dist < min_max[3] - min_max[2]:
        dist = min_max[3] - min_max[2]
    dist = dist * 1.25
    
    min_rect = [(min_max[0]+min_max[1])*0.5-dist*0.5, 
               (min_max[2]+min_max[3])*0.5-dist*0.5, 
               dist, dist]
    
    val2 = [0, 0, 0, 0]
    for i in range(12):
        val2[0] += val[114+i*2]
        val2[1] += val[115+i*2]
        val2[2] += val[138+i*2]
        val2[3] += val[139+i*2]
    
    angle = np.arctan2(val2[3]-val2[1], val2[2]-val2[0])
    return angle, min_rect

def get_cos_sine_matrix(angle, size=128):
    cos_val = np.cos(angle)
    sin_val = np.sin(angle)
    matrix = [0, 0, 0, 0, 0, 0]
    matrix[0] = cos_val
    matrix[4] = cos_val
    matrix[3] = sin_val
    matrix[1] = -sin_val
    matrix[2] = (size + size*sin_val - size*cos_val)*0.5
    matrix[5] = (size - (size*sin_val + size*cos_val))*0.5    
    return matrix

def get_left_right_eyes(landmarks):
    left_right_eye = [0, 0, 0, 0]
    for i in range(12):
        left_right_eye[0] += landmarks[114+i*2]
        left_right_eye[1] += landmarks[115+i*2]
        left_right_eye[2] += landmarks[138+i*2]
        left_right_eye[3] += landmarks[139+i*2]
    left_right_eye[0] /= 12
    left_right_eye[1] /= 12
    left_right_eye[2] /= 12
    left_right_eye[3] /= 12
    return left_right_eye

def calc_landmark_with_angle(output, angle):
    matrix = get_cos_sine_matrix(angle)
    val = np.zeros((244,))
    Dst = np.zeros((244,))
    frame = np.zeros((1, 128, 128))

    for i in range(113):
        pos0 = (coeff[2 * i] + output[2 * i]) * 64.0
        pos1 = (coeff[2 * i + 1] + output[2 * i + 1]) * 64.0
        Dst[2*i] = pos0
        Dst[2*i+1] = pos1
        frame[0, int(pos1), int(pos0)] = 255

    # Symmetry adjustments
    for i in range(3):
        if Dst[213+2*i] > Dst[225-2*i] or \
           (Dst[213+2*i-1] - Dst[225-2*i-1])**2 + (Dst[213+2*i] - Dst[225-2*i])**2 < 0.1:
            Dst[225-2*i] = (Dst[225-2*i] + Dst[213+2*i]) * 0.5
            temp = (Dst[225-2*i-1] + Dst[213+2*i-1]) * 0.5
            Dst[225-2*i-1] = temp
            Dst[213+2*i-1] = temp
            Dst[213+2*i] = Dst[225-2*i]
    
    # Rotate landmarks according to angle
    for i in range(113):
        pos0 = Dst[2*i]
        pos1 = Dst[2*i+1]
        val[2*i] = (matrix[1] * pos1) + (pos0 * matrix[0]) + matrix[2]
        val[2*i+1] = (matrix[4] * pos1) + (pos0 * matrix[3]) + matrix[5]
    
    return val

def calc_landmark(Dst, angle):
    return calc_landmark_with_angle(Dst, angle)

def get_landmark128(img):
    input_blob = img.reshape(1, 1, 128, 128).astype(np.float32)  # NCHW
    out = ort_session.run([outputs[0].name, outputs[1].name], input_feed={input_name: input_blob})
    return out[0]