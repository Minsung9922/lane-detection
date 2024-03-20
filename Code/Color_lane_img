import cv2
import numpy as np

# 이미지를 컬러로 읽어들입니다.
image_color = cv2.imread('./img.png', cv2.IMREAD_COLOR)
height, width = image_color.shape[:2]

# 사다리꼴 모양의 포인트를 정의합니다.
pts = np.array([[1400, 800], [0, 800], [width//2-250, 500], [width//2+50, 500]], dtype=np.int32)

# 사다리꼴 모양의 마스크를 생성합니다.
mask = np.zeros((height, width), dtype=np.uint8)
cv2.fillPoly(mask, [pts], (255, 255, 255))

# 마스크를 3채널로 확장합니다.
mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

# 마스크와 이미지를 결합합니다.
image_masked = cv2.bitwise_and(image_color, mask_3d)

cv2.imwrite('Masked Image.png', image_masked)

# 결합된 이미지를 그레이 스케일로 변환합니다.
image_gray = cv2.cvtColor(image_masked, cv2.COLOR_BGR2GRAY)

# 임계값을 설정하고 이진화 처리를 합니다.
threshold_value = 195
_, image_binary = cv2.threshold(image_gray, threshold_value, 255, cv2.THRESH_BINARY)

# 이진화된 영상을 보여줍니다.
cv2.imshow('Binary Image', image_binary)
cv2.imwrite('Binary Image.png', image_binary)
cv2.waitKey(0)
cv2.destroyAllWindows()
