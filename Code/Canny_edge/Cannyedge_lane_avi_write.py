import cv2
import numpy as np

# 비디오 파일을 엽니다.
cap = cv2.VideoCapture('./dataset/[mix]ORY_20170219_110743_D.avi')

# 비디오의 FPS와 프레임의 크기를 가져옵니다.
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 시작과 끝 시간에 해당하는 프레임 번호를 계산합니다.
start_time = 24  # 시작 시간 (초)
end_time = 28    # 끝 시간 (초)
start_frame_number = int(fps * start_time)
end_frame_number = int(fps * end_time)

# 비디오 저장을 위한 설정
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('processed_video.avi', fourcc, fps, (width, height), isColor=False)

current_frame = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 처리할 프레임 범위 내에 있는지 확인합니다.
    if current_frame >= start_frame_number and current_frame <= end_frame_number:
        # Canny edge detection을 적용합니다.
        edges = cv2.Canny(frame, 170, 300)
        edges_uint8 = np.uint8(edges)

        # 사다리꼴 모양의 마스크를 생성하고 적용합니다.
        pts = np.array([[660, 400], [0, 400], [235, 230], [390, 230]], dtype=np.int32)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        image_masked_edges = cv2.bitwise_and(edges_uint8, edges_uint8, mask=mask)

        # 처리된 프레임을 비디오 파일에 씁니다.
        out.write(image_masked_edges)

    current_frame += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
