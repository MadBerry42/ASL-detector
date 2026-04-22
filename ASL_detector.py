import numpy as np
import os
import mediapipe as mp
import cv2
from matplotlib import pyplot as plt
import time

#------------------ Keypoints using MP holistic ----------------------------------------------------
# Access webcam: Set up a video capture and loop through frames
cap = cv2.VideoCapture(0)                     # Accessing webcam (VideoCapture object number 0)
while cap.isOpened():                         # Doublecheck that the webcam is opened
    # Read Feed
    ret, frame = cap.read()                   # return value, image from the webcam = cap.read()
    # Show feed to the screen
    cv2.imshow('OpenCV Feed', frame)          # cv2.imshow('Frame name', frame)

    # Getting out of the loop
    if cv2.waitKey(10) & 0xFF == ord('q'):    # wait until a key is pressed and if the key is q, get out of the loop
        break

# Once it breaks, release the webcame and close the frame
cap.release()
cv2.destroyAllWindows()


