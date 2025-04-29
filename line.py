import cv2
import numpy as np
cap = cv2.VideoCapture(0)
prev_message = "I don't see the line"

while True:
    ret, frame = cap.read()
    #frame = frame[60:120, 0:160]
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
                prev_message = "Turn Right" # actual motor code + assign var
            if cx < 400 and cx > 240 :
                prev_message = "On Track!" # actual motor code + assign var
            if cx <=240 :
                prev_message = "Turn Left" # actual motor code + assign var
            print(prev_message)
            cv2.circle(frame, (cx,cy), 5, (255,0,0), -1)
            cv2.drawContours(frame, c, -1, (0,255,0), 1)
    else :
        # if line stops getting detected, turn unti l
        if prev_message == "Turn Right":
            print(prev_message) # actual motor code
        if prev_message == "On Track!":
            print(prev_message) # actual motor code
        if prev_message == "Turn Left":
            print(prev_message) # actual motor code
        else:
            print(prev_message)
    cv2.imshow("Mask",binary_image)
    cv2.imshow("Frame",frame)
    if cv2.waitKey(1) & 0xff == ord('q'):   # 1 is the time in ms
        break
cap.release()
cv2.destroyAllWindows()

