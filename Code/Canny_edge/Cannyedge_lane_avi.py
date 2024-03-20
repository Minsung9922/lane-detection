import cv2
import numpy as np

# 비디오 파일을 엽니다.
cap = cv2.VideoCapture('./dataset/[mix]ORY_20170219_110743_D.avi')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # 비디오의 끝에 도달했거나 읽기 오류

    height, width = frame.shape[:2]

    # Canny edge detection을 적용합니다.
    edges = cv2.Canny(frame, 170, 300)

    # 에지의 두께를 두껍게 하기 위해 딜레이션을 적용합니다.
    kernel = np.ones((3, 3), np.uint8)  # 딜레이션을 위한 커널, 크기를 조정하여 두께 조절 가능
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)  # 딜레이션 적용

    # 에지만 초록색으로 표시하는 새로운 컬러 이미지를 생성합니다.
    edge_color = np.zeros_like(frame)
    edge_color[edges_dilated != 0] = [0, 255, 0]

    # 사다리꼴 모양의 마스크를 생성하고 적용합니다.
    pts = np.array([[660, 400], [0, 400], [235, 230], [390, 230]], dtype=np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    # 마스크를 3채널로 확장하고, 마스크와 에지 컬러 이미지를 결합합니다.
    mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    masked_edge_color = cv2.bitwise_and(edge_color, mask_3d)

    # 원본 이미지와 마스크된 에지 컬러 이미지를 결합합니다.
    output_image = cv2.add(frame, masked_edge_color)

    # 이진화된 영상을 보여줍니다.
    cv2.imshow('Binary Image', output_image )

    if cv2.waitKey(15) & 0xFF == ord('q'):  # 30ms 대기 후 다음 프레임으로, 'q'를 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()
