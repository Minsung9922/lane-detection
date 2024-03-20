import cv2
import numpy as np

# 비디오 파일을 엽니다.
cap = cv2.VideoCapture('./dataset/[mix]ORY_20170219_110743_D.avi')

# 비디오의 FPS(초당 프레임 수)와 너비 및 높이를 검색합니다.
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 시작 시간과 종료 시간에 대한 프레임 번호를 계산합니다.
start_frame = int(24 * fps)
end_frame = int(28 * fps)

# 처리된 프레임을 저장하기 위해 비디오 작성기 객체를 생성합니다.
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_color222.avi', fourcc, fps, (width, height))

# 프레임 번호를 초기화합니다.
frame_number = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # 비디오의 끝이나 읽기 오류

    # 현재 프레임 번호가 시작 및 종료 프레임 내에 있는지 확인합니다.
    if start_frame <= frame_number <= end_frame:
        # 사다리꼴 모양의 점을 정의합니다.
        pts = np.array([[700, 400], [0, 400], [width // 2 - 250, 250], [width // 2 + 50, 250]], dtype=np.int32)

        # 사다리꼴 모양 마스크를 생성합니다.
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], (255, 255, 255))

        # 마스크를 3채널로 확장합니다.
        mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # 마스크와 이미지를 결합합니다.
        image_masked = cv2.bitwise_and(frame, mask_3d)

        # 결합된 이미지를 그레이스케일로 변환합니다.
        image_gray = cv2.cvtColor(image_masked, cv2.COLOR_BGR2GRAY)

        # 임계값을 설정하고 이진화를 적용합니다.
        threshold_value = 190
        _, image_binary = cv2.threshold(image_gray, threshold_value, 255, cv2.THRESH_BINARY)

        # 이진화된 이미지에서 흰색 픽셀의 위치를 찾아 원본 이미지에서 해당 위치를 초록색으로 변경합니다.
        frame[image_binary == 255] = [0, 255, 0]

        # 수정된 컬러 이미지를 저장합니다.
        out.write(frame)

    # 프레임 번호를 업데이트합니다.
    frame_number += 1

    if frame_number > end_frame:
        break  # 종료 시간 이후에 루프를 종료

cap.release()
out.release()
cv2.destroyAllWindows()
