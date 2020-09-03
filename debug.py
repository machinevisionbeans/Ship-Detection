import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Load image
filename = "/home/qdthk/Desktop/HaiPhong/5_25_mp4/19257.png"
inputImg = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
width = 1920
height = 1080

f = cv2.resize(inputImg, (width, height), interpolation=cv2.INTER_AREA)
#
# ===== I. Gray-scale morphological reconstruction
#
B = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 9))
# print( B )

# opening and closing map
g = cv2.morphologyEx(f, cv2.MORPH_OPEN, B)
h = cv2.morphologyEx(f, cv2.MORPH_CLOSE, B)

# compute intensity foreground saliency map
max_iters = 10
gi = g
hi = h
ogmr = np.zeros((height, width), np.uint8)
cgmr = np.ones((height, width), np.uint8) * 255
for it in range(max_iters):
    gi = cv2.dilate(gi, B)
    gdi = cv2.min(gi, f)
    ogmr = cv2.max(ogmr, gdi)

    hi = cv2.erode(hi, B)
    hdi = cv2.max(hi, f)
    cgmr = cv2.min(cgmr, hdi)

f_f = np.float32(f) / 255.0
ogmr_f = np.float32(ogmr) / 255.0
cgmr_f = np.float32(cgmr) / 255.0

ifsm_bf = cv2.pow(f_f - ogmr_f, 1)
ifsm_df = cv2.pow(cgmr_f - f_f, 1)

# compute background contrast saliency map
cgmr_open = cv2.morphologyEx(cgmr, cv2.MORPH_OPEN, B)
ogmr_close = cv2.morphologyEx(ogmr, cv2.MORPH_CLOSE, B)

bcsm_bf = cv2.pow(cgmr_f - np.float32(cgmr_open) / 255.0, 1)
bcsm_df = cv2.pow(np.float32(ogmr_close) / 255.0 - ogmr_f, 1)

# compute fused saliency map
min1, max1, iloc1, aloc1 = cv2.minMaxLoc(ifsm_bf)
min2, max2, iloc2, aloc2 = cv2.minMaxLoc(bcsm_bf)

ifsm_bf_norm = (ifsm_bf - min1) / (max1 - min1)
bcsm_bf_norm = (bcsm_bf - min2) / (max2 - min2)

sm_f = cv2.pow(cv2.multiply(ifsm_bf_norm, bcsm_bf_norm), 0.5)
sm_i = np.uint8(sm_f * 255)
ret, sm_bin = cv2.threshold(sm_i, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# avg, stddev = cv2.meanStdDev( sm_f )
# thresh = (avg + 15.0 * stddev) * 255
# ret, sm_bin = cv2.threshold( sm_i, thresh, 255, cv2.THRESH_BINARY )

# ====== Use it to remove small white regions
# B1 = cv2.getStructuringElement( cv2.MORPH_ELLIPSE, (5,5) )
# sm_bin = cv2.morphologyEx( sm_bin, cv2.MORPH_CLOSE, B1 )

# Preprocessing image
# fig1, axs = plt.subplots(3, 3, constrained_layout=True)
# fig1.suptitle( 'morphological images' )
# axs[0, 0].imshow( cv2.cvtColor(f, cv2.COLOR_BGR2RGB) )
# axs[0, 0].set_title( 'input image' )
# axs[0, 1].imshow( cv2.cvtColor(g, cv2.COLOR_BGR2RGB) )
# axs[0, 1].set_title( 'opening image' )
# axs[0, 2].imshow( cv2.cvtColor(h, cv2.COLOR_BGR2RGB) )
# axs[0, 2].set_title( 'closing image' )
#
# axs[1, 0].imshow( cv2.cvtColor(ogmr, cv2.COLOR_BGR2RGB) )
# axs[1, 0].set_title( 'OGMR' )
# axs[1, 1].imshow( cv2.cvtColor(cgmr, cv2.COLOR_BGR2RGB) )
# axs[1, 1].set_title( 'CGMR' )
# axs[1, 2].imshow( cv2.cvtColor(ifsm_bf_norm, cv2.COLOR_BGR2RGB) )
# axs[1, 2].set_title( 'IFSM' )
#
# axs[2, 0].imshow( cv2.cvtColor(bcsm_bf_norm, cv2.COLOR_BGR2RGB) )
# axs[2, 0].set_title( 'BCSM' )
# axs[2, 1].imshow( cv2.cvtColor(sm_f, cv2.COLOR_BGR2RGB) )
# axs[2, 1].set_title( 'Fused SM' )
# axs[2, 2].imshow( cv2.cvtColor(sm_bin, cv2.COLOR_BGR2RGB) )
# axs[2, 2].set_title( 'Binary SM' )
#
#
# x = np.arange(0, width, 1)
# y = np.arange(0, height, 1)
# X, Y = np.meshgrid(x, y)
#
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# # surf1 = ax.plot_surface(X, Y, inputImg, cmap=cm.coolwarm, linewidth=0, antialiased=False)
# surf2 = ax.plot_surface(X, Y, sm_f, cmap=cm.coolwarm, linewidth=0, antialiased=False)
#
# plt.show()


#
# ===== II. Contour description of TIR Ship based on Eigenvalue Analysis and Shape Constraint
#

# Find large eigen value map
xKernel = np.array([1, -1], dtype="float")
yKernel = xKernel.T

sm_x = cv2.filter2D(sm_f, -1, xKernel)
sm_y = cv2.filter2D(sm_f, -1, yKernel)
sm_xx = cv2.filter2D(sm_x, -1, xKernel)
sm_xy = cv2.filter2D(sm_x, -1, yKernel)
sm_yx = cv2.filter2D(sm_y, -1, xKernel)
sm_yy = cv2.filter2D(sm_y, -1, yKernel)

kSigma = (2, 2)
kSize = (13, 13)
sm_xx = cv2.GaussianBlur(sm_xx, kSize, sigmaX=kSigma[0], sigmaY=kSigma[1], borderType=cv2.BORDER_CONSTANT)
sm_xy = cv2.GaussianBlur(sm_xy, kSize, sigmaX=kSigma[0], sigmaY=kSigma[1], borderType=cv2.BORDER_CONSTANT)
sm_yx = cv2.GaussianBlur(sm_yx, kSize, sigmaX=kSigma[0], sigmaY=kSigma[1], borderType=cv2.BORDER_CONSTANT)
sm_yy = cv2.GaussianBlur(sm_yy, kSize, sigmaX=kSigma[0], sigmaY=kSigma[1], borderType=cv2.BORDER_CONSTANT)

lamdaLage = 0.5 * (sm_xx + sm_yy + cv2.sqrt(cv2.pow(sm_xx - sm_yy, 2) + 4 * sm_xy * sm_yx))
minv, maxv, minl, maxl = cv2.minMaxLoc(lamdaLage)
# print( (minv, maxv) )
lamdaLage = (lamdaLage - minv) / (maxv - minv)
# EXPERIMENT IT: Thresholding by Otsu or by zero thresh
thresh, lamdaLage_bin = cv2.threshold(np.uint8(lamdaLage * 255), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Find contour map of binary fused saliency map
B1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
sm_bin_dilate = cv2.dilate(sm_bin, B1)
contours, hierachy = cv2.findContours(sm_bin_dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

staeam_thresh = 0.2094
staem_valid_contours = []
for contId in range(len(contours)):
    x, y, w, h = cv2.boundingRect(contours[contId])
    staem_roi = lamdaLage[y:y + h, x:x + w]
    staem_mask = lamdaLage_bin[y:y + h, x:x + w]
    avg = cv2.meanStdDev(staem_roi, mask=staem_mask)
    if avg[0] > staeam_thresh:
        staem_valid_contours.append(contours[contId])
    else:
        sm_bin_dilate[y:y + h, x:x + w] *= 0  # np.zeros((h, w), np.uint8)

sm_bin_staem = cv2.erode(sm_bin_dilate, B1)
# sm_bin_staem = cv2.min( sm_bin_dilate, sm_bin )


#
# ===== III. Shape constraint verification
#
contours, hierachy = cv2.findContours(sm_bin_staem, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
detected_cont = []

minArea = 9
minStatisticArea = 30

for contId in range(len(contours)):
    contour = contours[contId]
    area = cv2.contourArea(contour)
    if area < minArea:
        continue
    if area > minStatisticArea:
        per = cv2.arcLength(contour, True)
        ellipse = cv2.fitEllipse(contour)
        majorAxis = np.float32(ellipse[1][1])
        minorAxis = np.float32(ellipse[1][0])
        ratio_mami = majorAxis / minorAxis
        compactness = per * per / area
        boundRect = cv2.minAreaRect(contour)
        rectangularity = area / (boundRect[1][0] * boundRect[1][1])

        if (ratio_mami > 1.1) and (ratio_mami < 8.8) and (compactness > 11.28) and (compactness < 118.62) and (
                rectangularity > 0.38) and (rectangularity < 0.88):
            detected_cont.append(contour)
    else:
        detected_cont.append(contour)

retImg = cv2.cvtColor(f, cv2.COLOR_GRAY2RGB)
# for i in range( len(contours) ):
cv2.drawContours(retImg, detected_cont, -1, (0, 255, 0), 2)

# fig2, axs = plt.subplots(2, 2, constrained_layout=True)
# fig2.suptitle( 'Structure Tensor Map' )
# axs[0, 0].imshow( cv2.cvtColor(lamdaLage, cv2.COLOR_BGR2RGB) )
# axs[0, 0].set_title( 'STAEM map' )
# axs[0, 1].imshow( cv2.cvtColor(sm_bin_dilate, cv2.COLOR_BGR2RGB) )
# axs[0, 1].set_title( 'Dilated binary saliency map' )
# axs[1, 0].imshow( cv2.cvtColor(sm_bin_staem, cv2.COLOR_BGR2RGB) )
# axs[1, 0].set_title( 'Connected region whose STAEM > Thr*' )
# axs[1, 1].imshow( retImg )
# axs[1, 1].set_title( 'Final detection result' )
# plt.show()
cv2.imshow("Structure Tensor Map", lamdaLage)
cv2.imshow("Dilated binary saliency map", sm_bin_dilate)
cv2.imshow("Connected region whose STAEM > Thr", sm_bin_staem)
cv2.imshow("Detect result", retImg)
cv2.waitKey()
