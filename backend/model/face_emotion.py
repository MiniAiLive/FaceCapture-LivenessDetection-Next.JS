import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import os

class EmotionDetector:
    def __init__(self):
        self.model = self._build_model()
        self.model.load_weights(os.path.join(os.path.dirname(__file__), "emotion/emotion.h5"))
        self.face_cascade = cv2.CascadeClassifier(
            os.path.join(os.path.dirname(__file__), "emotion/default.xml"))
        self.emotion_dict = {
            0: "Angry", 
            1: "Disgusted", 
            2: "Fearful", 
            3: "Happy", 
            4: "Neutral", 
            5: "Sad", 
            6: "Surprised"
        }
        
    def _build_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(7, activation='softmax'))
        return model
    
    def detect_emotions(self, image):
        """Detect emotions for all faces in an image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        emotions = []
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = self.model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            print(maxindex)
            emotions.append(self.emotion_dict[maxindex])
            
        return emotions