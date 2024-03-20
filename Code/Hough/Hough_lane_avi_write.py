import cv2
import numpy as np

# 비디오 파일을 엽니다.
cap = cv2.VideoCapture('./dataset/[mix]ORY_20170219_110743_D.avi')

# 비디오의 FPS와 프레임의 크기를 가져옵니다.
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 24초부터 28초까지의 프레임 번호를 계산합니다.
start_frame = int(24 * fps)
end_frame = int(28 * fps)

# 비디오 작성을 위한 VideoWriter 객체를 초기화합니다.
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('hough_to_4_seconds.avi', fourcc, fps, (width, height))

current_frame = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # 비디오의 끝에 도달했거나 읽기 오류

    # 현재 프레임 번호가 24초부터 28초 사이인지 확인합니다.
    if start_frame <= current_frame <= end_frame:
        # 여기에 프레임 처리 코드 삽입
        edges = cv2.Canny(frame, 200, 300)
        pts = np.array([[660, 400], [0, 400], [235, 230], [390, 230]], dtype=np.int32)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        image_masked_edges = cv2.bitwise_and(edges, edges, mask=mask)
        lines = cv2.HoughLinesP(image_masked_edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=100)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # 해당 프레임을 비디오 파일에 씁니다.
        out.write(frame)

    # 프레임 번호를 업데이트합니다.
    current_frame += 1

cap.release()
out.release()
cv2.destroyAllWindows()
