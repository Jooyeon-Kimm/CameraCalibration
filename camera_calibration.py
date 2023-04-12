# https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
import numpy as np
import cv2 as cv
import glob

def fileWrite():
    f = open("C:/Users/joooo/Desktop/images/보정결과.txt", 'w')
    data = "ret\n"
    f.write(data)
    data = str(ret)
    f.write(data)

    data = "\n\ncameraMatrix\n"
    f.write(data)
    data = str(mtx)
    f.write(data)

    data = "\n\ndistCoeffs\n"
    f.write(data)
    data = str(dist)
    f.write(data)

    data = "\n\nrvecs\n"
    f.write(data)
    data = str(rvecs)
    f.write(data)

    data = "\n\ntvecs\n"
    f.write(data)
    data = str(tvecs)
    f.write(data)

    data = "\n\ntotal error\n"
    f.write(data)
    data = str(mean_error / len(objpoints))
    f.write(data)

    f.close()


def printCameraCalibration():
    print("ret")
    print(ret)

    print("mtx")
    print(mtx)

    print("distCoeffs")
    print(dist)

    print("rvecs") # 회전벡터
    print(rvecs)

    print("tvecs") # 이동벡터
    print(tvecs)


# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((4*6,3), np.float32)
objp[:,:2] = np.mgrid[0:4,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# images = glob.glob('C:/Users/joooo/Desktop/images/*.jpg')
# for fname in images:
img = cv.imread('C:/Users/joooo/Desktop/images/20.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Find the chess board corners
ret, corners = cv.findChessboardCorners(gray, (4, 6), None)

# If found, add object points, image points (after refining them)
if ret == True:
    objpoints.append(objp)
    corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
    imgpoints.append(corners2)

"""
# Draw and display the corners
cv.drawChessboardCorners(img, (4,6), corners2, ret)
img2 = cv.resize(img, (1000,1000))

cv.imshow('img', img2)
cv.waitKey(0)
cv.destroyAllWindows()
"""

# camera calibration
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
printCameraCalibration()


# undistortion
h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
x, y, w, h = roi
dst = dst[y:y + h, x:x + w]
cv.imwrite('C:/Users/joooo/Desktop/images/calibresult2.jpg', dst)

# 원래영상 화면에 띄우기
img1 = cv.resize(img, (500,500))
cv.imshow('original', img1)

# 보정영상 화면에 띄우기
img2 = cv.resize(dst, (500,500))

cv.imshow('calibrated', img2)
cv.waitKey(0)
cv.destroyAllWindows()

# reprojection error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    mean_error += error
print("total error: {}".format(mean_error / len(objpoints)))

fileWrite()






