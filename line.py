import cv2
import numpy as np
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, binary_image = cv2.threshold(imgray, 200, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary_image, 1, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0 :
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] !=0 :
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            print("CX : "+str(cx)+"  CY : "+str(cy))
            if cx >= 400 :
                print("Turn Right")
            if cx < 400 and cx > 240 :
                print("On Track!")
            if cx <=240 :
                print("Turn Left")
            cv2.circle(frame, (cx,cy), 5, (255,0,0), -1)
            cv2.drawContours(frame, c, -1, (0,255,0), 1)
    else :
        print("I don't see the line")
    cv2.imshow("Mask",binary_image)
    cv2.imshow("Frame",frame)
    if cv2.waitKey(1) & 0xff == ord('q'):   # 1 is the time in ms
        break
cap.release()
cv2.destroyAllWindows()

