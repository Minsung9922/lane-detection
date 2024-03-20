import cv2
import numpy as np

# 비디오 파일을 엽니다.
cap = cv2.VideoCapture('./dataset/[mix]ORY_20170219_110743_D.avi')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # 비디오의 끝에 도달했거나 읽기 오류

    # Canny 에지 검출
    edges = cv2.Canny(frame, 200, 300)

    # 마스크 생성
    height, width = frame.shape[:2]
    pts = np.array([[660, 400], [0, 400], [235, 230], [390, 230]], dtype=np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    # 마스크 적용
    image_masked_edges = cv2.bitwise_and(edges, edges, mask=mask)

    # 허프 변환을 사용한 직선 검출
    lines = cv2.HoughLinesP(image_masked_edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=100)

    # 검출된 직선 그리기
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # 결과 영상 표시
    cv2.imshow('Detected Lines', frame)

    if cv2.waitKey(15) & 0xFF == ord('q'):  # 30ms 대기 후 다음 프레임으로, 'q'를 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()
