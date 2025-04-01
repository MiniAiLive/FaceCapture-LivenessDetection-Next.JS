import numpy as np
import cv2 as cv
import time
from onnx_align.onnx_align import get_landmark, get_angle
from onnx_landmark.onnx_landmark import get_landmark128, calc_landmark, get_left_right_eyes
from onnx_property.onnx_property import get_property, get_age_gender
from onnx_liveness.onnx_liveness import get_liveness
from onnx_detectface.onnx_detectface import detect_face, face_check, crop_image

def property_inference(img, bdboxes):
    img_cp = img
    age_info = []
    gender_info = []
    liveness_info = []
    angle_info = []
    start_time = time.time()
    crop_imgs = []
    for i in range(bdboxes.shape[0]):
        bdbox = bdboxes[i]
        width = bdbox[1]-bdbox[0]
        height = bdbox[3]-bdbox[2]
        delta = width-height
        if delta > 0:
            bdbox[2] -= delta / 2
            bdbox[3] += delta / 2
        else:
            bdbox[0] += delta / 2
            bdbox[1] -= delta / 2
        
        faceRect = [int(bdbox[0]), int(bdbox[2]), int(bdbox[1]-bdbox[0]) + 1, int(bdbox[3]-bdbox[2]) + 1]
        
        width, height = faceRect[2], faceRect[3]
        faceRectEx = [int(faceRect[0]-width*0.1), int(faceRect[1]-height*0.1), int(width*1.2), int(height*1.2)]
        # print("faceRectEx", faceRectEx)
        crop_img = crop_image(img, faceRectEx[0], faceRectEx[1], faceRectEx[2], faceRectEx[3]) #img[int(faceRectEx[1]):int(faceRectEx[1] + faceRectEx[3])+1, int(faceRectEx[0]):int(faceRectEx[0] + faceRectEx[2])+1, :]
        crop_img = cv.resize(crop_img, (64, 64))
        crop_img = cv.cvtColor(crop_img, cv.COLOR_BGR2GRAY)
        
        crop_img = (crop_img - 128.0)/255.0
        crop_img = crop_img.astype(np.float32)
        output = get_landmark(crop_img)
        end_time = time.time()
        execution_time = (end_time - start_time)*1000
        # print(f"get_landmark took {execution_time} miliseconds to execute.")

        Dst = output.reshape(-1, )
        angle, min_rect = get_angle(Dst, faceRectEx)
        (h, w) = img.shape[:2]
        center = (int(min_rect[0]+min_rect[2]/2), int(min_rect[1]+min_rect[3]/2))
        M = cv.getRotationMatrix2D(center, np.degrees(angle), 1)
        rotated = cv.warpAffine(img, M, (w, h))
        end_time = time.time()
        execution_time = (end_time - start_time)*1000
        # print(f"getRotationMatrix2D took {execution_time} miliseconds to execute.")
        crop_img = crop_image(rotated, min_rect[0], min_rect[1], min_rect[2], min_rect[3]) #rotated[int(min_rect[1]):int(min_rect[1] + min_rect[3])+1, int(min_rect[0]):int(min_rect[0] + min_rect[2])+1, :]
        crop_img = cv.resize(crop_img, (128, 128))
        crop_img = cv.cvtColor(crop_img, cv.COLOR_BGR2GRAY)
        crop_img = (crop_img - 128.0)/255.0
        crop_img = crop_img.astype(np.float32)
        
        output = get_landmark128(crop_img)
        landmkark113 = calc_landmark(output.reshape(-1,), angle)
        end_time = time.time()
        execution_time = (end_time - start_time)*1000
        # print(f"calc_landmark took {execution_time} miliseconds to execute.")
        new_landmarks = landmkark113
        for i in range(113):
            new_landmarks[2*i] = new_landmarks[2*i] * min_rect[2] / 128.0 + min_rect[0]
            new_landmarks[2*i+1] = new_landmarks[2*i+1] * min_rect[3] / 128.0 + min_rect[1]

        left_right_eye = get_left_right_eyes(landmkark113)

        distance = np.sqrt((left_right_eye[0] - left_right_eye[2])**2 + (left_right_eye[1] - left_right_eye[3])**2)

        angle = np.arctan2(left_right_eye[3]-left_right_eye[1],left_right_eye[2]-left_right_eye[0])

        center = [int((left_right_eye[0] + left_right_eye[2]) / 2),int((left_right_eye[1] + left_right_eye[3]) / 2)]

        size = int(distance * 3.52)

        ratio = ((2 * (1 + (distance * 0.5 * 1.5))) * 3.5) / 320.0
        ratio2 = distance / 64.0
        
        (h, w) = img.shape[:2]
        M = cv.getRotationMatrix2D(center, np.degrees(angle), 1)
        rotated = cv.warpAffine(img, M, (w, h))

        left_pos = (int(center[0]-0.5*size), int(center[1]))

        crop_img = crop_image(rotated, left_pos[0], int(center[1]-size/2), size, size) #rotated[int(center[1]-size/2):int(center[1]+size/2+1), left_pos[0]:left_pos[0]+size+1 , :]
        
        crop_img2 = crop_image(rotated, center[0]-int(ratio*160), int(center[1]-128*ratio),  int(320*ratio), int(320*ratio))

        crop_img3 = crop_image(rotated, center[0]-int(ratio2*64), int(center[1]-44*ratio2),  int(128*ratio), int(144*ratio))

        crop_img = cv.resize(crop_img, (60, 60))
        output = get_property(crop_img.reshape(1, 60, 60, 3).transpose(0, 3, 1, 2))
        
        age, gender = get_age_gender(output)
        end_time = time.time()
        execution_time = (end_time - start_time)*1000
        # print(f"get_age_gender took {execution_time} miliseconds to execute.")

        age_info.append(float(age))
        gender_info.append(float(gender))
        angle_info.append(float(np.degrees(angle)))

        crop_img2 = cv.resize(crop_img2, (320, 320))
        output = get_liveness(crop_img2.reshape(1, 320, 320, 3).transpose(0, 3, 1, 2))
        end_time = time.time()
        execution_time = (end_time - start_time)*1000
        # print(f"get_liveness took {execution_time} miliseconds to execute.")

        liveness_info.append(float(output))

        crop_img3 = cv.resize(crop_img3, (128, 144))
        crop_img3 = cv.cvtColor(crop_img3, cv.COLOR_BGR2GRAY)

        # Draw landmarks on the image
        for i in range(0, len(new_landmarks), 2):
            x = int(new_landmarks[i])
            y = int(new_landmarks[i + 1])
            cv.circle(img_cp, (x, y), 2, (0, 255, 0), -1)  # Draw a green circle at each landmark point
        crop_img = crop_image(img_cp, int(bdbox[0]), int(bdbox[2]), int(bdbox[1]-bdbox[0]) + 1, int(bdbox[3]-bdbox[2]) + 1)
        crop_imgs.append(crop_img)
    for i in range(bdboxes.shape[0]):
        bdbox = bdboxes[i]
        img_cp = cv.rectangle(img_cp, (int(bdbox[0]), int(bdbox[2])), (int(bdbox[1]), int(bdbox[3])), (255, 0, 0),
                           thickness=2)

    return age_info, gender_info, liveness_info, angle_info

def detect_inference(img):
    return detect_face(img)
