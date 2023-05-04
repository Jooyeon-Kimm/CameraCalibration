import cv2
import numpy as np

def ncc_match(template, image):
    # 템플릿과 이미지를 grayscale로 변환
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 템플릿과 이미지의 크기를 가져옴
    th, tw = template_gray.shape[:2]
    ih, iw = image_gray.shape[:2]

    # 템플릿과 이미지의 평균값을 계산
    t_mean = np.mean(template_gray)
    i_mean = np.mean(image_gray)

    # 템플릿과 이미지의 분산값을 계산
    t_var = np.sum((template_gray - t_mean) ** 2)
    i_var = np.sum((image_gray - i_mean) ** 2)

    # normalized cross-correlation 계산
    corr = np.zeros((ih - th, iw - tw), dtype=np.float32)
    for y in range(ih - th):
        for x in range(iw - tw):
            patch = image_gray[y:y + th, x:x + tw]
            p_mean = np.mean(patch)
            p_var = np.sum((patch - p_mean) ** 2)
            corr[y, x] = np.sum((patch - p_mean) * (template_gray - t_mean)) / np.sqrt(p_var * t_var)

    # 이미지에서 템플릿이 매칭된 좌표를 찾음
    max_loc = np.unravel_index(np.argmax(corr), corr.shape)
    top_left = max_loc[::-1]
    bottom_right = (top_left[0] + tw, top_left[1] + th)

    # 이미지에 사각형을 그리고 매칭된 이미지를 반환
    result = cv2.rectangle(image.copy(), top_left, bottom_right, (0, 0, 255), 2)

    # correlation 이미지 생성
    corr_norm = cv2.normalize(corr, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    corr_color = cv2.applyColorMap(corr_norm, cv2.COLORMAP_JET)
    corr_gray = cv2.cvtColor(corr_norm, cv2.COLOR_GRAY2BGR)
    corr_gray = cv2.cvtColor(corr_gray, cv2.COLOR_BGR2GRAY)
    return corr_color, corr_gray, result

# 이미지와 템플릿을 불러옴
image = cv2.imread('./pen.jpg')
template = cv2.imread('./pen_temp1.jpg')

# template matching 실행
corr_color, corr_gray, result = ncc_match(template, image)

# 결과물 출력
cv2.imshow('Template', template)
cv2.imshow('Correlation_Color', corr_color)
cv2.imshow('Correlation_Gray', corr_gray)
cv2.imshow('Template Matching Result', result)
cv2.waitKey()
cv2.destroyAllWindows()