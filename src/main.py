import cv2
import numpy as np
import time
WINDOW_NAME = 'win'
DISP_DENOM = 2

cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorKNN(
    history=50, dist2Threshold=500.0, detectShadows=False)

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
# cv2.startWindowThread() # Not an option on pi with jupyter.
# Should definately try if running on other platforms.
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)//DISP_DENOM)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)//DISP_DENOM)
display_img = np.empty((height, width, 3), dtype=np.uint8)
cv2.createTrackbar('History', WINDOW_NAME,
                   fgbg.getHistory(), 500,
                   fgbg.setHistory)
cv2.createTrackbar('dist2Threshold', WINDOW_NAME,
                   int(fgbg.getDist2Threshold()), 400,
                   fgbg.setDist2Threshold)

frame_cnt = 0
while(True):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    display_img = cv2.resize(
        frame*np.atleast_3d(fgmask.astype(np.bool)),
        (width, height), interpolation=cv2.INTER_AREA)
    cv2.getTrackbarPos('History', WINDOW_NAME)
    cv2.getTrackbarPos('dist2Threshold', WINDOW_NAME)
    cv2.imshow(WINDOW_NAME, display_img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    frame_cnt += 1
    msg = 'History {0}, dist2Threshold {1}'.format(
              fgbg.getHistory(), fgbg.getDist2Threshold())
    print("\r", msg, end="")

cap.release()
cv2.waitKey(1)
cv2.destroyAllWindows()
cv2.waitKey(1)
