import numpy as np
import cv2

def ncc_match(template, image):
    # 템플릿과 이미지를 grayscale로 변환
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 템플릿과 이미지의 크기를 가져옴
    th, tw = template_gray.shape[:2]
    ih, iw = image_gray.shape[:2]

    # 이미지와 템플릿의 윈도우 사이즈를 계산
    win_h, win_w = int(ih * 0.1), int(iw * 0.1)

    # 이미지와 템플릿의 평균값을 계산
    image_mean = cv2.blur(image_gray, (win_h, win_w))
    template_mean = cv2.blur(template_gray, (win_h, win_w))

    # 이미지와 템플릿의 표준편차를 계산
    image_std = cv2.sqrt(np.float32(cv2.absdiff(image_gray, image_mean)))
    template_std = cv2.sqrt(np.float32(cv2.absdiff(template_gray, template_mean)))

    # 이미지와 템플릿의 정규화된 버전을 계산
    image_norm = (image_gray - image_mean) / (image_std + 1e-5)
    template_norm = (template_gray - template_mean) / (template_std + 1e-5)

    # 템플릿과 이미지의 cross-correlation을 계산
    result = cv2.matchTemplate(image_norm, template_norm, cv2.TM_CCOEFF_NORMED)

    # 템플릿과 일치하는 영역을 찾음
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    x, y = max_loc

    # 템플릿이 일치하는 영역을 사각형으로 표시
    cv2.rectangle(image, (x, y), (x + tw, y + th), (0, 0, 255), 2)

    return image

image = cv2.imread('./pen.jpg')
template = cv2.imread('./pen_temp1.jpg')
result = ncc_match(template, image)

cv2.imshow('result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
