import cv2
import numpy as np

# 비디오 파일을 엽니다.
cap = cv2.VideoCapture('./dataset/[mix]ORY_20170219_110743_D.avi')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # 비디오의 끝에 도달했거나 읽기 오류

    height, width = frame.shape[:2]

    # 여기서는 낮은 임계값을 50, 높은 임계값을 150으로 설정합니다.
    edges = cv2.Canny(frame, 170, 300)

    edges_uint8 = np.uint8(edges)

    pts = np.array([[660, 400], [0, 400], [235, 230], [390, 230]], dtype=np.int32)

    mask = np.zeros((height, width), dtype=np.uint8)

    cv2.fillPoly(mask, [pts], 255)

    image_masked_edges = cv2.bitwise_and(edges_uint8, edges_uint8, mask=mask)

    # 이진화된 영상을 보여줍니다.
    cv2.imshow('Binary Image', image_masked_edges)

    if cv2.waitKey(15) & 0xFF == ord('q'):  # 30ms 대기 후 다음 프레임으로, 'q'를 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()
