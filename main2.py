import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import ship_detection as sd

cap = cv2.VideoCapture( "data/seq1.mp4" )

while True:
	ret, frame = cap.read()
	if( ret == False ):
		print ("Failed to read frame")
		break

	gray = cv2.cvtColor( frame, cv2.COLOR_RGB2GRAY )

	detectList = sd.detect_ship( gray )
	cv2.drawContours( frame, detectList, -1, (0, 255, 0), 2 )
	cv2.imshow( "window", frame )
	cv2.waitKey( 10 )