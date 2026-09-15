import cv2
import numpy as np
import time

def nothing(x):
    pass

 
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: could not open webcam")
    exit()

# windows
cv2.namedWindow("bars", cv2.WINDOW_NORMAL)
cv2.resizeWindow("bars", 420, 220)
cv2.namedWindow("Harry's Cloak", cv2.WINDOW_NORMAL)
cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)

# trackbars (defaults tuned for blue cloth)
cv2.createTrackbar("lower_hue", "bars", 94, 180, nothing)
cv2.createTrackbar("upper_hue", "bars", 126, 180, nothing)
cv2.createTrackbar("lower_saturation", "bars", 80, 255, nothing)
cv2.createTrackbar("upper_saturation", "bars", 255, 255, nothing)
cv2.createTrackbar("lower_value", "bars", 50, 255, nothing)
cv2.createTrackbar("upper_value", "bars", 255, 255, nothing)


#background capture 

print("Please move out of the frame for 2 seconds so background can be captured...")
time.sleep(1.0)
background = None
for i in range(30):
    ret, frame = cap.read()
    if ret:
        background = frame.copy()
    cv2.waitKey(30)

if background is None:
    print("Error: couldn't capture background")
    cap.release()
    exit()

print("Background captured. Put your blue cloth in frame. Press 'b' to re-capture background anytime.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame not received")
        break

    # read trackbars
    lh = cv2.getTrackbarPos("lower_hue", "bars")
    uh = cv2.getTrackbarPos("upper_hue", "bars")
    ls = cv2.getTrackbarPos("lower_saturation", "bars")
    us = cv2.getTrackbarPos("upper_saturation", "bars")
    lv = cv2.getTrackbarPos("lower_value", "bars")
    uv = cv2.getTrackbarPos("upper_value", "bars")

    lower_hsv = np.array([lh, ls, lv], dtype=np.uint8)
    upper_hsv = np.array([uh, us, uv], dtype=np.uint8)

    # convert to HSV and smooth
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_blur = cv2.GaussianBlur(hsv, (7,7), 0)

    # mask for blue range
    mask = cv2.inRange(hsv_blur, lower_hsv, upper_hsv)

    # denoise
    mask = cv2.medianBlur(mask, 7)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # keep only the largest connected component — helps remove speckles
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        biggest = max(contours, key=cv2.contourArea)
        new_mask = np.zeros_like(mask)
        cv2.drawContours(new_mask, [biggest], -1, 255, thickness=cv2.FILLED)
        # smooth edges a bit
        new_mask = cv2.dilate(new_mask, np.ones((5,5),np.uint8), iterations=1)
        mask = new_mask

    # Inverse mask to keep non-cloak parts of current frame
    mask_inv = cv2.bitwise_not(mask)

    # Create result images
    res1 = cv2.bitwise_and(frame, frame, mask=mask_inv)         # current frame except cloak
    res2 = cv2.bitwise_and(background, background, mask=mask)  # background where cloak is

    final = cv2.addWeighted(res1, 1, res2, 1, 0)

    cv2.imshow("Harry's Cloak", final)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('b'):
        # recapture background (useful if lighting changed)
        print("Re-capturing background... please move out of frame")
        time.sleep(1.0)
        for i in range(30):
            ret, bgf = cap.read()
            if ret:
                background = bgf.copy()
            cv2.waitKey(30)
        print("Background updated.")

# cleanup
cap.release()
cv2.destroyAllWindows()
