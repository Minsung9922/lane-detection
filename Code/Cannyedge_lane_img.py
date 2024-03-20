import cv2
import numpy as np

# 이미지를 컬러로 읽어들입니다.
image_color = cv2.imread('./img.png', cv2.IMREAD_COLOR)

# 이미지를 그레이스케일로 변환합니다.
image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

# 케니 엣지 검출을 적용합니다.
edges = cv2.Canny(image_gray, 10, 180)  # 임계값을 조절하여 굵기를 설정

# 선의 굵기를 조절합니다.
edges_thick = cv2.dilate(edges, None, iterations=2)

# 빨간색으로 표시할 부분을 생성합니다.
red_edges = np.zeros_like(image_color)
red_edges[edges_thick != 0] = [0, 0, 255]  # 빨간색

# 빨간색 엣지 이미지를 저장합니다.
cv2.imwrite('red_edges.png', red_edges)
