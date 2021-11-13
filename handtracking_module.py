import cv2
import time
import mediapipe as mp

class HandTracking():
    def __init__(self, mode = False, max_hands = 2, model_complex = 1, detection_confidence = 0.5, track_confidence = 0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.model_complex = model_complex
        self.detection_confidence = detection_confidence
        self.track_confidence = track_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            self.mode,
            self.max_hands,
            self.model_complex,
            self.detection_confidence,
            self.track_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils

    def track_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        
        return img

    
    def find_position(self, img, hand_number = 0, draw = True):
        landmarks_list = []

        if self.results.multi_hand_landmarks:

            my_hand = self.results.multi_hand_landmarks[hand_number]

            for id, landmark in enumerate(my_hand.landmark):
                height, width, channel = img.shape
                center_x, center_y = int(landmark.x * width), int(landmark.y * height)
                landmarks_list.append([id, center_x, center_y])

                if draw:
                    cv2.circle(img, (center_x, center_y), 15, (255, 0, 255), cv2.FILLED)

        return landmarks_list



        