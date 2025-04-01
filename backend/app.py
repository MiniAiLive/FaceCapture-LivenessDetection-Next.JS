import sys
sys.path.append('./model')

from dotenv import load_dotenv
load_dotenv('../.env')  # Load from .env file

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image

from model.face_detect import detect_inference, property_inference
from model.face_emotion import EmotionDetector
from model.face_maskdetect import MaskDetector

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "https://face-capture-liveness-detection-web-next-js.vercel.app/"}})  # Allow Vercel domain)

# Initialize detectors
emotion_detector = EmotionDetector()
mask_detector = MaskDetector()

def base64_to_image(base64_string):
    img_data = base64.b64decode(base64_string)
    img = Image.open(BytesIO(img_data))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

def process_age(age_float):
    base_age = int(round(age_float))
    return f"{base_age-2} ~ {base_age+2}"

def process_liveness(liveness_score):
    return "Real" if liveness_score > 0.7 else "Fake"

def process_gender(gender_code):
    return "Male" if gender_code == 0 else "Female"

def crop_faces(image, bdboxes):
    cropped_faces = []
    for box in bdboxes:
        left, right, top, bottom = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        face_img = image[top:bottom, left:right]
        cropped_faces.append(face_img)
    return cropped_faces

@app.route('/api/detect', methods=['POST'])
def detect_faces():
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({"error": "No image provided"}), 400
            
        image = base64_to_image(data['image'])
        img_height, img_width = image.shape[:2]
        
        # Run model inference
        bdboxes = detect_inference(image)
        age_info, gender_info, liveness_info, angle_info = property_inference(image, bdboxes)
        
        # Validate inference results
        if not all(len(arr) == len(bdboxes) for arr in [age_info, gender_info, liveness_info, angle_info]):
            return jsonify({"error": "Inference output dimension mismatch"}), 500
            
        face_count = len(bdboxes)
        
        # Crop faces and run additional detectors
        cropped_faces = crop_faces(image, bdboxes)
        face_images = [image_to_base64(face) for face in cropped_faces]
        emotions = emotion_detector.detect_emotions(image)
        mask_results = mask_detector.detect_masks(image, bdboxes)
        
        # Build structured response
        structured_response = []
        for i in range(face_count):
            face_data = {
                "faceIndex": i+1,
                "face": face_images[i],
                "age": process_age(age_info[i]),
                "gender": process_gender(gender_info[i]),
                "liveness": process_liveness(liveness_info[i]),
                "emotion": emotions[i] if i < len(emotions) else "Unknown",
                "mask": mask_results[i] if i < len(mask_results) else {"status": "Unknown", "confidence": 0},
                "angle": float(angle_info[i]),
                "boundingBox": {
                    "left": float(bdboxes[i][0]),
                    "top": float(bdboxes[i][2]),
                    "width": float(bdboxes[i][1] - bdboxes[i][0]),
                    "height": float(bdboxes[i][3] - bdboxes[i][2])
                }
            }
            structured_response.append(face_data)
        
        return jsonify({
            "faces": structured_response,
            "imageInfo": {
                "width": img_width,
                "height": img_height
            },
            "faceCount": face_count
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3001, debug=True)
