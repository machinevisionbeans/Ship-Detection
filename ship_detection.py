import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import wasserstein_distance
from sklearn.cluster import MeanShift
from sklearn import linear_model


# ----------------------------------------------------
#
# Ship Detection based on morphological reconstruction
#
# Paper: Thermal infrared small ship detection in sea clutter based on
# morphological reconstruction and multi-feature analysis
# ----------------------------------------------------
def detect_ship(gray):
    width = 854
    height = 480
    f = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)

    # ===== I. Gray-scale morphological reconstruction
    B = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 9))

    # opening and closing map
    g = cv2.morphologyEx(f, cv2.MORPH_OPEN, B)
    h = cv2.morphologyEx(f, cv2.MORPH_CLOSE, B)
    tophat = cv2.morphologyEx(f, cv2.MORPH_TOPHAT, B)

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

    # ===== II. Contour description of TIR Ship based on Eigenvalue Analysis and Sahpe Constraint

    # Find lage eigen value map
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
    print((minv, maxv))
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

    # ===== III. Shape constraint verification
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
            minorAzis = np.float32(ellipse[1][0])
            ratio_mami = majorAxis / minorAzis
            compactness = per * per / area
            boundRect = cv2.minAreaRect(contour)
            rectangularity = area / (boundRect[1][0] * boundRect[1][1])

            if (ratio_mami > 1.1) and (ratio_mami < 8.8) and (compactness > 11.28) and (compactness < 118.62) and (
                    rectangularity > 0.38) and (rectangularity < 0.88):
                detected_cont.append(contour)
        else:
            detected_cont.append(contour)

    retImg = cv2.cvtColor(f, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(retImg, detected_cont, -1, (0, 255, 0), 2)
    return sm_bin_staem, retImg


# ----------------------------------------------------------
#
#	Ship detection based on outlier detection algorithm
#
#	Dan's idea
# ----------------------------------------------------------
def detect_ship_v2(gray):
    #
    # ======== I. Preprocessing
    #
    h, w = gray.shape

    # Noise reduction
    gaussImg = cv2.GaussianBlur(gray, (5, 5), sigmaX=2, sigmaY=2, borderType=cv2.BORDER_CONSTANT)

    # Estimate foreground texture
    B = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    f = cv2.morphologyEx(gaussImg, cv2.MORPH_OPEN, B)
    diff = gaussImg - f
    diffF = np.float32(diff) / 255.0

    #
    # ======== II. Statistic on foreground blobs to classify between ships and noise
    #

    # Find foreground blobs
    thresh, binDiff = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
    B1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binDiff = cv2.erode(binDiff, B1)
    binDiffF = np.float32(binDiff) / 255.0
    contours, _ = cv2.findContours(binDiff, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    avgEnergyList = []
    percentile70List = []
    oriList = []
    histId = 0
    validConts = []
    histList = []
    for id in range(len(contours)):
        cont = contours[id]
        rect = cv2.boundingRect(cont)
        patch = diffF[rect[1]: (rect[1] + rect[3]), rect[0]: (rect[0] + rect[2])]
        mask = binDiffF[rect[1]: (rect[1] + rect[3]), rect[0]: (rect[0] + rect[2])]
        maskI = binDiff[rect[1]: (rect[1] + rect[3]), rect[0]: (rect[0] + rect[2])]

        # Skip blobs whose area is smaller than 5x5
        if cv2.contourArea(cont) < 9:
            mask = mask * 0
            binDiffF[rect[1]: (rect[1] + rect[3]), rect[0]: (rect[0] + rect[2])] = mask
            continue
        validConts.append(cont)

        # estimate average signal energy inside the blob
        # energy = cv2.multiply(patch**2, mask)
        # avgEnergy = cv2.sumElems( energy )
        hp, wp = patch.shape
        horizontalAvgEnergy = 0
        for row in range(hp):
            rowPatch = patch[row, :]
            rowMaskI = maskI[row, :]
            avgEnergy = cv2.mean(rowPatch * rowPatch, rowMaskI)
            horizontalAvgEnergy = horizontalAvgEnergy + avgEnergy[0]

        # avgEnergy = cv2.mean( patch**2, maskI )
        # horizontalAvgEnergy = avgEnergy[0] * float(hp)
        avgEnergyList.append(horizontalAvgEnergy)

        # find value than that there are 30% number of pixels have their intensity higher
        p1 = patch.flatten()
        p1.sort()
        partition = 0
        for j in range(len(p1)):
            if p1[j] > 0:
                partition = j
                break
        p2 = p1[partition: len(p1)]
        idxPercentile70 = int(0.9 * len(p2))
        percentile70 = p2[idxPercentile70]
        percentile70List.append(percentile70)

    #
    # =========== III. Detect outlier using linear regression
    #
    x_train = np.array(avgEnergyList)
    x_train = x_train.reshape(-1, 1)
    y_train = np.array(percentile70List)
    y_train = y_train.reshape(-1, 1)

    ransac = linear_model.RANSACRegressor()
    ransac.residual_threshold = 0.2
    ransac.max_trials = 1000
    reg = ransac.fit(x_train, y_train)
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)
    x_test = np.arange(0, 2, 0.1)
    x_test = x_test.reshape(-1, 1)
    y_pred = reg.predict(x_test)

    detectList = []
    for id in range(len(validConts)):
        if outlier_mask[id]:
            # if avgEnergyList[id] > 1.2:
            detectList.append(validConts[id])

    return detectList


# ----------------------------------------------------------
#
#	Ship detection based on Multiscale Gray Difference
#
#	Paper: Infrared small-target detection using multiscale gray difference weighted image entropy
# ----------------------------------------------------------
def mgdFilter(gray, maxSize):
    grayF = np.float32(gray) / 255.0
    filtKmax = cv2.blur(grayF, (maxSize, maxSize))
    D = np.zeros(gray.shape, np.float32)
    for k in range(3, maxSize, 2):
        filtK = cv2.blur(grayF, (k, k))
        Dk = cv2.pow(filtK - filtKmax, 2)
        D = cv2.max(D, Dk)

    return D


def localEntropy(gray, blkSize):
    grayF = np.float32(gray) / 255.0
    h, w = grayF.shape
    for ii in range(h):
        for jj in range(w):
            if (grayF[ii, jj] == 0):
                grayF[ii, jj] = grayF[ii, jj] + 0.000001

    kernel = np.ones((blkSize, blkSize), np.float32)

    logGray = cv2.log(grayF)
    grayLogGray = cv2.multiply(grayF, logGray)
    filtGrayLogGray = cv2.filter2D(grayLogGray, -1, kernel)

    filtGray = cv2.filter2D(grayF, -1, kernel)
    filtGray = filtGray + 0.000001
    logFiltGray = cv2.log(filtGray)

    lE = logFiltGray - filtGrayLogGray / filtGray

    return lE