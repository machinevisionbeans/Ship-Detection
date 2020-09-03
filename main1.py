import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import ship_detection as sd

cap = cv2.VideoCapture( "/home/brain/Desktop/ship-detection-cpp/data1.mp4")

while True:
    ret, frame = cap.read()
    if (ret == False):
        print("Failed to read frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    D = sd.mgdFilter(gray, 15)
    lE = sd.localEntropy(gray, 7)
    nwie = cv2.multiply(D,lE)

    minVal, maxVal, _, _ = cv2.minMaxLoc(nwie)
    nwief = (nwie - minVal) / (maxVal - minVal) *255.0
    nwieF = nwief.astype(np.uint8)
    _, nwieF = cv2.threshold(nwieF, 10, 255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)

    cv2.imshow("output", nwieF)
    cv2.waitKey(10)
