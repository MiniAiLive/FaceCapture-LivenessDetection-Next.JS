import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os

class MaskDetector:
    def __init__(self):
        # Load face detection model
        prototxtPath = os.path.join(os.path.dirname(__file__), "mask_detect/deploy.prototxt")
        weightsPath = os.path.join(os.path.dirname(__file__), "mask_detect/mask_detector.caffemodel")
        self.face_net = cv2.dnn.readNet(prototxtPath, weightsPath)
        
        # Load mask detection model
        modelPath = os.path.join(os.path.dirname(__file__), "mask_detect/mask_detector.model")
        self.mask_model = load_model(modelPath)
    
    def detect_masks(self, image, bdboxes):
        """Detect masks for faces in the image using provided bounding boxes"""
        mask_results = []
        
        for box in bdboxes:
            startX, endX, startY, endY = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            
            # Extract face ROI
            face = image[startY:endY, startX:endX]
            
            # Preprocess face for mask detection
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)
            
            # Predict mask
            (mask_prob, no_mask_prob) = self.mask_model.predict(face)[0]
            
            # Determine result
            label = "Mask" if mask_prob > no_mask_prob else "No Mask"
            confidence = max(mask_prob, no_mask_prob)
            
            mask_results.append({
                "status": label,
                "confidence": float(confidence)
            })
            
        return mask_results