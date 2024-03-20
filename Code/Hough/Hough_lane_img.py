import cv2
import numpy as np

# 이미지를 그레이스케일로 읽어들입니다.(케니에지 적용한 이미지사용)
canny_img = cv2.imread('./masked_edges.png', cv2.IMREAD_GRAYSCALE)

# 컬러 이미지로 다시 읽어서 결과 직선을 그리기 위한 준비(원본 영상 사용 또는 에지 검출된 이미지 사용)
image_color = cv2.imread('./masked_edges.png', cv2.IMREAD_COLOR)

# 허프 변환 적용
# minLineLength - 직선으로 간주될 최소 길이
# maxLineGap - 같은 직선으로 간주될 최대 간격
lines = cv2.HoughLinesP(canny_img, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=300)

# 검출된 직선 그리기
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image_color, (x1, y1), (x2, y2), (0, 255, 0), 2)

# 결과 이미지 저장
cv2.imwrite('./hough_lines_result.png', image_color)

# 결과 이미지 보기
cv2.imshow('Detected Lines', image_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
