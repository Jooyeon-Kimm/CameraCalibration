import cv2 as cv
from matplotlib import pyplot as plt

# 왼쪽 이미지와 오른쪽 이미지 로드
left_image = cv.imread('im0.png', cv.IMREAD_GRAYSCALE)
right_image = cv.imread('im1.png', cv.IMREAD_GRAYSCALE)

# StereoSGBM 객체 생성
stereo = cv.StereoSGBM_create(minDisparity=0, numDisparities=128, blockSize=1)

# Disparity 맵 계산
disparity = stereo.compute(left_image, right_image)

# 입력 이미지 출력
plt.subplot(121)
plt.imshow(left_image, cmap='gray')
plt.title('Left Image')
plt.axis('off') # 축 표시하지 않음

plt.subplot(122)
plt.imshow(right_image, cmap='gray')
plt.title('Right Image')
plt.axis('off')  # 축 표시하지 않음

plt.show()

# Disparity 맵 출력
plt.imshow(disparity, cmap='jet')
plt.title('Disparity Map')
plt.axis('off')
plt.show()

# Disparity 맵을 컬러 이미지로 변환
disparity_normalized = cv.normalize(disparity, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)  # 데이터 타입 변환
disparity_color = cv.applyColorMap(disparity_normalized, cv.COLORMAP_JET)

# 컬러 이미지로 저장
cv.imwrite('disparity_color.png', disparity_color)

