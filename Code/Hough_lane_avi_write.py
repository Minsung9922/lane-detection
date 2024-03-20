import cv2
import numpy as np

# Open the video file.
cap = cv2.VideoCapture('./dataset/[mix]ORY_20170219_110743_D.avi')

# Retrieve the FPS (frames per second), and the width and height of the video.
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Calculate the frame numbers for the start and end times.
start_frame = int(24 * fps)
end_frame = int(28 * fps)

# Create a video writer object to save the processed frames.
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))

# Initialize the frame number.
frame_number = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # End of video or read error

    # Check if the current frame number is within the start and end frames.
    if start_frame <= frame_number <= end_frame:
        # Define the points for the trapezoidal shape.
        pts = np.array([[700, 400], [0, 400], [width // 2 - 250, 250], [width // 2 + 50, 250]], dtype=np.int32)

        # Create the trapezoidal shape mask.
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], (255, 255, 255))

        # Extend the mask to 3 channels.
        mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Combine the mask with the image.
        image_masked = cv2.bitwise_and(frame, mask_3d)

        # Convert the combined image to grayscale.
        image_gray = cv2.cvtColor(image_masked, cv2.COLOR_BGR2GRAY)

        # Set the threshold and apply binarization.
        threshold_value = 190
        _, image_binary = cv2.threshold(image_gray, threshold_value, 255, cv2.THRESH_BINARY)

        # 이진화된 이미지에서 흰색 픽셀의 위치를 찾아 원본 이미지에서 해당 위치를 초록색으로 변경합니다.
        frame[image_binary == 255] = [0, 255, 0]

        # Save the modified binarized image.
        out.write(frame)

    # Update the frame number.
    frame_number += 1

    if frame_number > end_frame:
        break  # Exit the loop after the end time

cap.release()
out.release()
cv2.destroyAllWindows()
