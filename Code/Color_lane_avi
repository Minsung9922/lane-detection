import cv2
import numpy as np

# 비디오 파일을 엽니다.
cap = cv2.VideoCapture('./dataset/[mix]ORY_20170219_110743_D.avi')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # 비디오의 끝에 도달했거나 읽기 오류
    height, width = frame.shape[:2]

    # 사다리꼴 모양의 포인트를 정의합니다.
    pts = np.array([[700, 400], [0, 400], [width // 2 - 250, 250], [width // 2 + 50, 250]], dtype=np.int32)

    # 사다리꼴 모양의 마스크를 생성합니다.
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], (255, 255, 255))

    # 마스크를 3채널로 확장합니다.
    mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # 마스크와 이미지를 결합합니다.
    image_masked = cv2.bitwise_and(frame, mask_3d)

    # 결합된 이미지를 그레이 스케일로 변환합니다.
    image_gray = cv2.cvtColor(image_masked, cv2.COLOR_BGR2GRAY)

    # 임계값을 설정하고 이진화 처리를 합니다.
    threshold_value = 190
    _, image_binary = cv2.threshold(image_gray, threshold_value, 255, cv2.THRESH_BINARY)

    # 이진화된 영상을 보여줍니다.
    cv2.imshow('Binary Image', image_binary)

    if cv2.waitKey(15) & 0xFF == ord('q'):  # 30ms 대기 후 다음 프레임으로, 'q'를 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()
