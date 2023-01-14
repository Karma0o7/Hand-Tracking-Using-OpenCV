"""
Hand Tracking Module
This program uses Hand Tracking Module to track the hand and find the position of the landmarks
Made Using OpenCV and Mediapipe
"""


import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm


def main():
    point = int(input("Enter the point you want to track: "))
    if point > 21 and point < 0:
        print("Invalid Point")
        return
    prevTime = 0
    currTime = 0
    cap = cv2.VideoCapture(0)
    detector = htm.handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=True, handPoint=point)
        if len(lmList) != 0:
            print(lmList[point])
 
        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime
 
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)
 
        cv2.imshow("Image", cv2.flip(img, 1))
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break
 
 
if __name__ == "__main__":
    main()