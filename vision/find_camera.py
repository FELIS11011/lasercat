import cv2

for i in range(4):
    cap = cv2.VideoCapture(i)
    ok = cap.isOpened()
    got = False
    if ok:
        r, f = cap.read()
        got = r
    print(f"index {i}: opened={ok}  frame={got}")
    cap.release()