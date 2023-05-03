import cv2
import numpy as np

img = cv2.imread('./duck.jpg', cv2.IMREAD_GRAYSCALE) # 배경이미지, 흑백으로 불러오기
tmplt = cv2.imread('./duck_temp1.jpg', cv2.IMREAD_GRAYSCALE) # 찾을 이미지. 불러올때부터 흑백

w, h = tmplt.shape[::-1] # 타겟의 크기값을 변수에 할당

res = cv2.matchTemplate(img, tmplt, cv2.TM_CCOEFF_NORMED) # 코릴레이션 결과
threshold = 0.6 # 0~1의 값. 높으면 적지만 정확한 결과. 낮으면 많지만 낮은 정확도.
loc = np.where(res>=threshold) # res에서 threshold보다 큰 값만 취한다.

for pt in zip(*loc[::-1]):
    cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2) # 결과값에 사각형을 그린다
    cv2.rectangle(res, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)  # 결과값에 사각형을 그린다

cv2.imshow("template", tmplt)
cv2.imshow("image", img)
cv2.imshow("correlation_result", res)

cv2.waitKey(0)
cv2.destroyAllWindows()


