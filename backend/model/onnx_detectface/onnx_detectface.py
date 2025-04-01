import cv2 as cv
import numpy as np
import onnxruntime as ort
import os
from cryptography.fernet import Fernet

# Load encryption key from environment variable
MODEL_KEY = os.getenv("MODEL_KEY")
if not MODEL_KEY:
    raise ValueError("MODEL_KEY environment variable not set")

cipher_suite = Fernet(MODEL_KEY.encode())

# Paths to encrypted models
encrypted_detect_path = os.path.join(os.path.dirname(__file__), "detect_model3.onnx.enc")
encrypted_cls_path = os.path.join(os.path.dirname(__file__), "face_cls_model.onnx.enc")

# Load and decrypt the detection model
with open(encrypted_detect_path, "rb") as f:
    encrypted_data = f.read()
decrypted_detect = cipher_suite.decrypt(encrypted_data)
ort_session = ort.InferenceSession(decrypted_detect)
input_name = ort_session.get_inputs()[0].name  # 'data'
outputs = ort_session.get_outputs()[0].name    # 'fc1'

# Load and decrypt the classification model
with open(encrypted_cls_path, "rb") as f:
    encrypted_data = f.read()
decrypted_cls = cipher_suite.decrypt(encrypted_data)
ort_session_face_cls = ort.InferenceSession(decrypted_cls)
input_name_face_cls = ort_session_face_cls.get_inputs()[0].name  # 'data'
outputs_face_cls = ort_session_face_cls.get_outputs()[0].name    # 'fc1'

def crop_image(img, x, y, crop_w, crop_h):
    x = int(x)
    y = int(y)
    crop_w = int(crop_w)
    crop_h = int(crop_h)

    (h, w) = img.shape[:2]
    padding = [0, 0, 0, 0]
    if x < 0:
        padding[0] = -x
    if y < 0:
        padding[2] = -y
    
    if x+crop_w > w:
        padding[1] = x+crop_w - w
    if y+crop_h > h:
        padding[3] = y+crop_h - h
    
    crop_img = np.pad(img, ((padding[2], padding[3]), (padding[0], padding[1]), (0, 0)), "constant")
    return crop_img[y+padding[2]:y+padding[2]+crop_h+1, x+padding[0]:x+padding[0]+crop_w+1, :]

def nms_func(boxes, scores, iou_threshold=0.5):
    areas = (boxes[:, 1] - boxes[:, 0] + 0.001) * (boxes[:, 3] - boxes[:, 2] + 0.001)
    order = scores.argsort()[::-1]
    keep = []
    
    while np.size(order) > 1:
        i = order[0]
        keep.append(i)
        indices = []
        
        for j in range(1, np.size(order)):
            k = order[j]
            xx1 = max(boxes[i, 0], boxes[k, 0])
            yy1 = max(boxes[i, 2], boxes[k, 2])
            xx2 = min(boxes[i, 1], boxes[k, 1])
            yy2 = min(boxes[i, 3], boxes[k, 3])
            width = max(0.0, xx2 - xx1 + 0.001)
            height = max(0.0, yy2 - yy1 + 0.001)
            inter = width * height
            iou = inter / (areas[i] + areas[k] - inter)
            
            if iou <= iou_threshold:
                indices.append(j)
                
        if len(indices) == 0:
            break
        order = order[indices]
    return keep

def detect_face(img, conf_thres=0.5, cls_thres=0.5):
    size = max(img.shape)
    img_crop = crop_image(img, 0, 0, max(img.shape[:2]), max(img.shape[:2]))
    img_crop = cv.resize(img_crop, (320, 320))
    input_blob = img_crop.transpose((2, 0, 1)).reshape(1, 3, 320, 320).astype(np.float32)  # NCHW
    
    out = ort_session.run([outputs], input_feed={input_name: input_blob})
    output = out[0]
    indices = output[:, 0] > conf_thres
    output = output[indices]
    indices = nms_func(output[:, 1:5], output[:, 0])
    bdboxes = output[indices][:, 1:] * size
    
    new_bdboxes = []
    for i in range(bdboxes.shape[0]):
        bdbox = bdboxes[i]
        width = bdbox[1] - bdbox[0]
        height = bdbox[3] - bdbox[2]
        center = ((bdbox[0] + bdbox[1]) / 2, (bdbox[2] + bdbox[3]) / 2)
        size = width
        
        if size < height:
            size = height
        size = int(size)
        
        bdboxes[i][0] = int(center[0] - size/2)
        bdboxes[i][2] = int(center[1] - size/2)
        bdboxes[i][1] = bdboxes[i][0] + size
        bdboxes[i][3] = bdboxes[i][2] + size
        
        crop_img = crop_image(img, center[0]-size/2, center[1]-size/2, size, size)
        crop_img = cv.resize(crop_img, (64, 64))
        crop_img = cv.cvtColor(crop_img, cv.COLOR_BGR2GRAY)
        out = face_check(crop_img)
        
        if out[0, 1] > cls_thres:
            new_bdboxes.append([bdboxes[i][0], bdboxes[i][1], bdboxes[i][2], bdboxes[i][3]])
    
    return np.array(new_bdboxes)

def face_check(img):
    input_blob = img.reshape(1, 1, 64, 64).astype(np.float32)  # NCHW
    out = ort_session_face_cls.run([outputs_face_cls], input_feed={input_name_face_cls: input_blob})
    return out[0]