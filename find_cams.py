import cv2

## Finding webcams' ids
print("Scanning for cameras...")
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"Camera found at index {i}")
        cap.release()
print("Scan ended.")