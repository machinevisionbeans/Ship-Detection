import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import ship_detection as sd

# =========== NWIE ship detector
cap = cv2.VideoCapture("../../data/seq1.mp4")

while True:
    ret, frame = cap.read()
    if (ret == False):
        print("Failed to read frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    D = sd.mgdFilter(gray, 15)
    lE = sd.localEntropy(gray, 7)
    nwie = cv2.multiply(D, lE)

    # threshold proposed by paper
    meanV, stdVal = cv2.meanStdDev(nwie)
    _, maxVal, _, _ = cv2.minMaxLoc(nwie)
    T = 0.05 * (maxVal - meanV[0]) + meanV[0]
    nwieBin = np.uint8(nwie > T) * 255

    # print(meanV)

    cv2.imshow("output", nwieBin)
    cv2.waitKey(10)

# =========== Debug 1 image
# img = cv2.imread( "/home/quantum/DandoWP/AntiShip/data/selected/good/1_506.png", cv2.IMREAD_GRAYSCALE )
# D = sd.mgdFilter( img, 25 )
# lE = sd.localEntropy( img, 7 )
# nwie = cv2.multiply(D, lE)
# meanV, stdVal = cv2.meanStdDev( nwie )
# _, maxVal, _, _ = cv2.minMaxLoc( nwie )
# T = 0.1 * (maxVal - meanV[0]) + meanV[0]
# nwieBin = np.uint8(nwie > T) * 255

# cv2.imshow( "input", img )
# cv2.imshow( "output", nwieBin)
# cv2.waitKey( 0 )


# --------------------------------------------------------------
#
#	VIDEO TO IMAGES
#
# --------------------------------------------------------------

# cap = cv2.VideoCapture( "/home/dando/Videos/milesight/2K100.avi" )
# c = 1
# while (True):
# 	ret, frame = cap.read()
# 	if( ret == False ):
# 		print ("Failed to read frame")
# 		break

# 	name = "/home/dando/Videos/milesight/frames_2K100/img_" + str(c) + ".png"
# 	# cv2.imwrite( name, frame )
# 	print( name )
# 	c = c + 1


# --------------------------------------------------------------
#
#	FLICK TEST
#
# --------------------------------------------------------------

# cap = cv2.VideoCapture( "/home/quantum/DandoWP/AntiShip/data/seq3_bigBoat.mp4" )

# prevGray = []
# pause = False
# rect = []
# intensityList = []
# while True:

# 	if pause == False:
# 		ret, frame = cap.read()
# 		if( ret == False ):
# 			print ("Failed to read frame")
# 			break
# 		gray = cv2.cvtColor( frame, cv2.COLOR_RGB2GRAY )
# 		grayf = np.float32( gray )
# 		kps = cv2.cornerHarris( grayf, 5, 3, 0.04 )
# 		frame[kps > 0.01*kps.max()] = [0, 0, 255]

# 	else:
# 		rect = cv2.selectROI( frame )
# 		pause = False

# 	if( rect != [] and rect[2] > 0 and rect[3] > 0):
# 		cv2.rectangle( frame, (int(rect[0]), int(rect[1])), (int(rect[0]+rect[2]), int(rect[1]+rect[3])), (0, 255, 0), 1 )
# 		patch = gray[int(rect[1]):int(rect[1]+rect[3]), int(rect[0]):int(rect[0]+rect[2])]
# 		meanV, _ = cv2.meanStdDev( patch )
# 		intensityList.append( meanV[0] )


# 	cv2.imshow( "window", frame )
# 	k = cv2.waitKey( 30 )
# 	if k == 32:
# 		pause = not pause

# 	# Flick display
# 	prevGray = gray

# if( len(intensityList) > 0 ):
# 	x = np.arange( 0, len(intensityList), 1 )
# 	plt.plot( intensityList )
# 	plt.ylabel( 'avg intensity' )
# 	plt.xlabel( 'time' )
# 	plt.show()