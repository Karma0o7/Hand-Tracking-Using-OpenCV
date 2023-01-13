import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm


def main():
    prevTime = 0
    currTime = 0
    cap = cv2.VideoCapture(0)
    detector = htm.handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[0])
 
        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime
 
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)
 
        cv2.imshow("Image", img)
        if cv2.waitKey(1) == ord('q'):
            break
 
 
if __name__ == "__main__":
    main()