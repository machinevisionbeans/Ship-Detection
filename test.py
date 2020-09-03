import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as st
from scipy.stats import wasserstein_distance
from sklearn.cluster import MeanShift
from sklearn import linear_model
from sklearn.neighbors import KernelDensity

filename = '/home/qdthk/Desktop/HaiPhong/5_25_mp4/20136.png'
histFileName = "histograms.txt"
FILE = open(histFileName, "w")

#
# ======== I. Preprocessing
#
inputImg = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
h, w = inputImg.shape

# Noise reduction
gaussImg = cv2.GaussianBlur(inputImg, (5, 5), sigmaX=2, sigmaY=2, borderType=cv2.BORDER_CONSTANT)

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

# Compute image gradient
Gx = cv2.Sobel(diffF, -1, 1, 0, ksize=3)
Gy = cv2.Sobel(diffF, -1, 0, 1, ksize=3)
Gxx = cv2.multiply(Gx, Gx)
Gxy = cv2.multiply(Gx, Gy)
Gyy = cv2.multiply(Gy, Gy)

avgEnergyList = []
percentile70List = []
entropies = []
oriList = []
histId = 0
validConts = []
histList = []
test_val = []
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

    # ======== estimate orientation of the blob
    # gxx = cv2.multiply( Gxx[rect[1] : (rect[1]+rect[3]), rect[0] : (rect[0]+rect[2])], mask )
    # gxy = cv2.multiply( Gxy[rect[1] : (rect[1]+rect[3]), rect[0] : (rect[0]+rect[2])], mask )
    # gyy = cv2.multiply( Gyy[rect[1] : (rect[1]+rect[3]), rect[0] : (rect[0]+rect[2])], mask )

    # gxxSum = cv2.sumElems( gxx )[0]
    # gxySum = cv2.sumElems( gxy )[0]
    # gyySum = cv2.sumElems( gyy )[0]
    # ori = math.pi + 0.5 * math.atan2(2*gxySum, gxxSum-gyySum)
    # if ori > math.pi:
    # 	ori = ori - 2*math.pi
    # if ori < 0:
    # 	ori = ori + math.pi
    # if ori > math.pi/2:
    # 	ori = math.pi - ori
    # oriList.append( ori )

    # ========== calculate patch histogram and local entropy
    hp, wp = patch.shape
    maxSize = max([hp, wp])
    hn = int((100.0 / float(maxSize)) * float(hp))
    wn = int((100.0 / float(maxSize)) * float(wp))
    patchN = cv2.resize(patch, (wn, hn), interpolation=cv2.INTER_AREA)
    maskIN = cv2.resize(maskI, (wn, hn), interpolation=cv2.INTER_AREA)
    hist = cv2.calcHist([patchN], [0], maskIN, [16], [0.0, 1.0])
    sumHist = cv2.sumElems(hist)
    hist = hist / sumHist[0]
    hist = hist.flatten()

    histList.append(hist)
    FILE.writelines(["%f\t" % val for val in hist])
    FILE.write("\r\n")
    histId = histId + 1

    entropy = 0
    for ii in range(len(hist)):
        if (hist[ii] > 0):
            entropy = entropy - hist[ii] * np.log2(hist[ii])
    entropies.append(np.exp(entropy))
    test_val.append([horizontalAvgEnergy, percentile70, entropy])

FILE.close()

#
# =========== III. Detect outlier using linear regression using RANSAC
#
x_train = np.array(avgEnergyList)
x_train = x_train.reshape(-1, 1)
y_train = np.array(percentile70List)
y_train = y_train.reshape(-1, 1)
z_train = np.array(entropies)
z_train = z_train.reshape(-1, 1)

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
        detectList.append(validConts[id])

xk = np.linspace(x_train.min(), x_train.max(), 1000)
yk = np.linspace(y_train.min(), y_train.max(), 1000)
XK, YK = np.meshgrid(xk, yk)
pos = np.vstack([YK.ravel(), XK.ravel()])
xy = np.vstack([y_train.ravel(), x_train.ravel()])
kernel = st.gaussian_kde(xy)
Z = np.reshape(kernel(pos).T, XK.shape)

#
# =========== IV. Display for debuging
#
fig, axes = plt.subplots(1, 2, constrained_layout=True)
axes[0].imshow(cv2.cvtColor(inputImg, cv2.COLOR_BGR2RGB))
axes[0].set_title('I')
axes[1].imshow(cv2.cvtColor(f, cv2.COLOR_GRAY2BGR))
axes[1].set_title('BgrI')

avgEnergyList = np.array(avgEnergyList)
percentile70List = np.array(percentile70List)
oriList = np.array(oriList)

x = np.arange(0, w, 1)
y = np.arange(0, h, 1)
X, Y = np.meshgrid(x, y)

fig = plt.figure()
ax = fig.gca(projection='3d')
surf2 = ax.plot_surface(X, Y, diff, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# fig, axes = plt.subplots(1, 2, constrained_layout=True)
gaussImg = cv2.cvtColor(gaussImg, cv2.COLOR_GRAY2RGB)
cv2.drawContours(gaussImg, detectList, -1, (0, 255, 0), 2)
# axes[0].imshow( cv2.cvtColor(gaussImg, cv2.COLOR_BGR2RGB) )
# axes[1].imshow( cv2.cvtColor(binDiffF, cv2.COLOR_BGR2RGB) )

fig = plt.figure()
plt.imshow(cv2.cvtColor(gaussImg, cv2.COLOR_BGR2RGB))

fig = plt.figure()
plt.scatter(x_train[inlier_mask], y_train[inlier_mask], color='yellowgreen', marker='.',
            label='Inliers')
plt.scatter(x_train[outlier_mask], y_train[outlier_mask], color='gold', marker='.',
            label='Outliers')
plt.plot(x_test, y_pred, color="blue", linewidth=2)

# ax = fig.gca(projection='3d')
# # plt.scatter(x_train[inlier_mask], y_train[inlier_mask], z_train[inlier_mask], color='yellowgreen', marker='.',
# #             label='Inliers')
# # plt.scatter(x_train[outlier_mask], y_train[outlier_mask], z_train[outlier_mask], color='gold', marker='.',
# #             label='Outliers')
# plt.scatter(x_train, y_train, z_train, color='yellowgreen', marker='.',
#             label='Inliers')
# ax.set_xlabel( 'x' )
# ax.set_ylabel( 'y' )
# ax.set_zlabel( 'z' )


plt.show()