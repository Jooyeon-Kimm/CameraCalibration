# Computer Vision
# https://git.ajou.ac.kr/givemebro/cv_3.git
# 두 장의 영상을 붙여 한 개의 영상을 만듬
# 1. 특징점이 적지 않은 곳을 촬영하고, 카메라를 중심으로 30도 정도 회전 시켜 (겹치는 부분이 영상의 반 정도가 되도록) 한 장을 더 촬영합니다.
# 2. 두 영상 각각을 <gray image>로 만들고, 특징점과 descriptor를 생성합니다.
# 3. Brute-force matcher를 이용하여 두 영상의 feature point 간의 match를 계산하고,  출력합니다. (drawMatches 함수 사용)
# 주의: Descriptor의 종류에 따라서 matching과정의 distance measure가 다릅니다.
#       적절한 것을 선택해야 합니다. (EX, ORB의 경우 cv.NORM_HAMMING, SURF의 경우에는 cv.NORM_L2)
# 4. match들을 이용하여 2번째 image의 perspective transform를 계산합니다. 이때 ransac을 사용하는 것이 좋습니다.
# 5. 두번째 image를 warp하여 큰 image를 만들고, 첫번째 image를 필요한 부분에 복사하여 합성된 영상을 생성하고 출력합니다.

import cv2
import numpy as np
DBG = False

""" ************** 왼쪽+중간 / 중간+오른쪽 영상 2개 추출하기 (시작) ************** """
# 왼쪽/오른쪽/중간 사진 읽기
imgL = cv2.imread('img_L3.jpg')
imgR = cv2.imread('img_R3.jpg')
imgM = cv2.imread('img_M3.jpg')

hL, wL = imgL.shape[:2]
hR, wR = imgR.shape[:2]
hM, wM = imgM.shape[:2]

# gray scale
grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
grayM = cv2.cvtColor(imgM, cv2.COLOR_BGR2GRAY)

# SIFT 디스크립터 추출기 생성
descriptor = cv2.SIFT_create()

# 각 영상에 대해 키 포인트와 디스크립터 추출
(kpsL, featuresL) = descriptor.detectAndCompute(imgL, None)
(kpsR, featuresR) = descriptor.detectAndCompute(imgR, None)
(kpsM, featuresM) = descriptor.detectAndCompute(imgM, None)

# assign
imgL_draw = cv2.drawKeypoints(imgL, kpsL, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
imgR_draw = cv2.drawKeypoints(imgR, kpsR, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
imgM_draw = cv2.drawKeypoints(imgM, kpsM, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# 좌/우/중간 이미지의 키 포인트 전시
if DBG:
    cv2.imshow('SIFT_L', imgL_draw)
    cv2.imshow('SIFT_R', imgR_draw)
    cv2.imshow('SIFT_M', imgM_draw)
    cv2.waitKey(0)

# BF 매칭기 생성 및 knn 매칭
matcher = cv2.DescriptorMatcher_create("BruteForce")
matches_LM = matcher.knnMatch(featuresM, featuresL, 2)
matches_MR = matcher.knnMatch(featuresR, featuresM, 2)

# 좌/우/중간 이미지의 매칭점 연결 전시
if DBG:
    res_LM = cv2.drawMatchesKnn(imgL, kpsL, imgM, kpsM, matches_LM, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    res_MR = cv2.drawMatchesKnn(imgM, kpsM, imgR, kpsR, matches_MR, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('KnnMatech + SIFT', res_LM)
    cv2.imshow('KnnMatech + SIFT', res_MR)
    cv2.waitKey(0)

# 좋은 매칭점 선별
good_matches_LM = []
good_matches_MR = []

for m in matches_LM:
    if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
        good_matches_LM.append((m[0].trainIdx, m[0].queryIdx))

for m in matches_MR:
    if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
        good_matches_MR.append((m[0].trainIdx, m[0].queryIdx))

# 좋은 매칭점이 4개 이상인 원근 변환행렬 구하기
if len(good_matches_LM) > 4:
    ptsL = np.float32([kpsL[i].pt for (i, _) in good_matches_LM])
    ptsM = np.float32([kpsM[i].pt for (_, i) in good_matches_LM])
    matrix, status = cv2.findHomography(ptsM, ptsL, cv2.RANSAC, 4.0)
    # 원근 변환행렬로 오른쪽과 사진을 원근 변환, 결과 이미지 크기는 사징 2장 크기
    imgLM = cv2.warpPerspective(imgM, matrix, (wM + wL, hR))
    # 왼쪽 사진을 원근 변환한 왼쪽 영역에 합성
    imgLM[0:hL, 0:wL] = imgL

# 좋은 매칭점이 4개 이상인 원근 변환행렬 구하기
if len(good_matches_MR) > 4:
    ptsM = np.float32([kpsM[i].pt for (i, _) in good_matches_MR])
    ptsR = np.float32([kpsR[i].pt for (_, i) in good_matches_MR])
    matrix, status = cv2.findHomography(ptsR, ptsM, cv2.RANSAC, 4.0)
    # 원근 변환행렬로 오른쪽과 사진을 원근 변환, 결과 이미지 크기는 사징 2장 크기
    imgMR = cv2.warpPerspective(imgR, matrix, (wM + wR, hR))
    # 왼쪽 사진을 원근 변환한 왼쪽 영역에 합성
    imgMR[0:hM, 0:wM] = imgM

else:
    panorama_LM = imgL
    panorama_MR = imgM
"""  ************** 왼쪽+중간 / 중간+오른쪽 영상 2개 추출하기 (완료) ************** """


"""  ************** 왼쪽+중간 / 중간+오른쪽 영상 2개 Alignment (시작) ************** """
# 각 영상에 대해 키 포인트와 디스크립터 추출
(kpsLM, featuresLM) = descriptor.detectAndCompute(imgLM, None)
(kpsMR, featuresMR) = descriptor.detectAndCompute(imgMR, None)

# assign
imgLM_draw = cv2.drawKeypoints(imgLM, kpsLM, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
imgMR_draw = cv2.drawKeypoints(imgMR, kpsMR, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


# 좌/우/중간 이미지의 키 포인트 전시
if DBG:
    cv2.imshow('SIFT_LM', imgLM_draw)
    cv2.imshow('SIFT_MR', imgMR_draw)
    cv2.waitKey(0)

# BF 매칭기 생성 및 knn 매칭
matcher = cv2.DescriptorMatcher_create("BruteForce")
matches_LMR = matcher.knnMatch(featuresMR, featuresLM, 2)

# 좌/우/중간 이미지의 매칭점 연결 전시
if DBG:
    res_LMR = cv2.drawMatchesKnn(imgLM, kpsLM, imgMR, kpsMR, matches_LMR, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('KnnMatech + SIFT', res_LMR)
    cv2.waitKey(0)

# 좋은 매칭점 선별
good_matches_LMR = []

for m in matches_LMR:
    if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
        good_matches_LMR.append((m[0].trainIdx, m[0].queryIdx))


# 좋은 매칭점이 4개 이상인 원근 변환행렬 구하기
if len(good_matches_LMR) > 4:
    ptsLM = np.float32([kpsLM[i].pt for (i, _) in good_matches_LMR])
    ptsMR = np.float32([kpsMR[i].pt for (_, i) in good_matches_LMR])
    matrix, status = cv2.findHomography(ptsMR, ptsLM, cv2.RANSAC, 4.0)
    # 원근 변환행렬로 오른쪽과 사진을 원근 변환, 결과 이미지 크기는 사징 2장 크기
    imgLMR = cv2.warpPerspective(imgMR, matrix, (wL + wM + wR, hR))
    # 왼쪽 사진을 원근 변환한 왼쪽 영역에 합성
    imgLMR[0:hM, 0:wL+wM] = imgLM

else:
    imgLMR = imgLM
""" ************** 왼쪽+중간 / 중간+오른쪽 영상 2개 Alignment (완료) ************** """

cv2.imshow("Image Left", imgL)
cv2.imshow("Image Middle", imgM)
cv2.imshow("Image Right", imgR)

cv2.imshow("Image Left+Middle", imgLM)
cv2.imshow("Image Middle+Right", imgMR)

cv2.imshow("Panorama_SIFT: imgLMR", imgLMR)
cv2.imwrite("Panorama_SIFT.jpg: imgLMR", imgLMR)
cv2.waitKey(0)
cv2.destroyAllWindows()