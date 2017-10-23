import cv2
import numpy as np


WINDOW_NAME = 'win'

cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.startWindowThread()
cv2.waitKey(1)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)//2)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)//2)
display_img = np.empty((height, width), dtype=np.uint8)
cv2.waitKey(1)

while(1):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    display_img = cv2.resize(fgmask, (width, height),
                             interpolation=cv2.INTER_AREA)
    cv2.imshow(WINDOW_NAME, display_img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.waitKey(1)
cv2.destroyAllWindows()
cv2.waitKey(1)
