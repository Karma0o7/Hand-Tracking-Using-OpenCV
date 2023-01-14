"""
Hand Tracking Module
This is hand tracking module which can be used to track the hand and find the position of the landmarks 
and draw the landmarks on the hand
Made Using OpenCV and Mediapipe 
"""


import cv2
import mediapipe as mp
import time
 
class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
 
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, 1,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
 
    def findHands(self, img, draw=True):
        """
        This function finds the hands in the image and draws the landmarks on the hand
        :param img: Image in which the hand is to be detected
        :param draw: Boolean value to draw the landmarks on the hand
        :return: Image with the landmarks drawn on the hand
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
 
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img
 
    def findPosition(self, img, handNo=0, draw=True, handPoint=None):
        """
        This function finds the position of the landmarks in the image
        :param img: Image in which the hand is to be detected
        :param handNo: Hand on which the landmarks are to be drawn
        :param draw: Boolean value to draw the landmarks on the hand
        :param handPoint: Point of the hand to be detected
        :return: List of the landmarks
        """
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw and id == handPoint:
                    cv2.circle(img, (cx, cy), 8, (255, 0, 255), cv2.FILLED)
        return lmList

 