import cv2
import numpy as np
import math

param_rho = 1
param_theta = 1
param_threshold = 100
param_minlen = 5
param_maxgap = 10

def detectLine(frame):
    """
    Process the given frame to detect and track the center of a white line.
    
    Args:
        frame (numpy.ndarray): The input frame from the webcam.
    
    Returns:
        lineCenter: A number between [-1, 1] denoting where the center of the line is relative to the frame.
        newFrame: Processed frame with the detected line marked using cv2.rectangle() and center marked using cv2.circle().
    """
    height, width, _ = frame.shape
#     lineCenter = np.interp(lineCenter, [0, width], [-1, 1])
    lineCenter = [0.0, 0.0]
    minPt = [width, height]
    maxPt = [0.0, 0.0]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dst = cv2.Canny(frame, 50, 200, None, 3)
    lines = cv2.HoughLines(dst, param_rho, param_theta * np.pi / 180, param_threshold, None, 0, 0)
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            if pt1[0] < minPt[0]:
                minPt[0] = pt1[0]
            if pt1[1] < minPt[1]:
                minPt[1] = pt1[1]
            if pt2[0] > maxPt[0]:
                maxPt[0] = pt2[0]
            if pt2[1] > maxPt[1]:
                maxPt[1] = pt2[1]
            cv2.line(frame, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
        minPt[0] = max(0, minPt[0])
        minPt[1] = max(0, minPt[1])
        maxPt[0] = min(width, maxPt[0])
        maxPt[1] = min(height, maxPt[1])
        print("Min Point:", minPt)
        print("Max Point:", maxPt)
        cv2.rectangle(frame, (int(minPt[0]), int(minPt[1])), (int(maxPt[0]), int(maxPt[1])), (0, 255, 0), 3)
        lineCenter[0] = (minPt[0] + maxPt[0])/2
        lineCenter[1] = (minPt[1] + maxPt[1])/2
        cv2.circle(frame, (int(lineCenter[0]), int(lineCenter[1])), 5, (255, 0, 0), 3)
    # lines_list =[]
    # lineCenter = 0.0
    # lines = cv2.HoughLinesP(
    #     dst, # Input edge image
    #     param_rho, # Distance resolution in pixels
    #     param_theta*np.pi/180, # Angle resolution in radians
    #     threshold=param_threshold, # Min number of votes for valid line
    #     minLineLength=param_minlen, # Min allowed length of line
    #     maxLineGap=param_maxgap # Max allowed gap between line for joining them
    # )
    # # Iterate over points
    # if lines is not None:
    #     for points in lines:
    #         # Extracted points nested in the list
    #         x1,y1,x2,y2=points[0]
    #         lineCenter += x1 + x2
    #         # Draw the lines joing the points
    #         # On the original image
    #         cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),2)
    #         # Maintain a simples lookup list for points
    #         lines_list.append([(x1,y1),(x2,y2)])
    #     lineCenter /= (len(lines_list)*2)
    #     _, width, _ = frame.shape
    #     lineCenter = np.interp(lineCenter, [0, width], [-1, 1])
    #     print("# Lines:", len(lines_list))
    return lineCenter, frame

def main():
    global param_rho
    global param_theta
    global param_threshold
    global param_minlen
    global param_maxgap

    cam = cv2.VideoCapture(1)  # Open webcam

    while cam.isOpened():
        ret, frame = cam.read()
        if not ret:
            break

        cv2.imshow('Original', frame)

        lineCenter, newFrame = detectLine(frame)
        # newFrame = detectLine(frame)

        cv2.imshow('Lines', newFrame)
        print("Line Center:", lineCenter)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('w'):
            param_rho += 1
        elif key == ord('s'):
            if param_rho > 1:
                param_rho -= 1 
            else:
                param_rho = 1
        elif key == ord('e'):
            param_theta += 1
        elif key == ord('d'):
            if param_theta > 1:
                param_theta -= 1 
            else:
                param_theta = 1
        elif key == ord('r'):
            param_threshold += 5
        elif key == ord('f'):
            if param_threshold > 5:
                param_threshold -= 1 
            else:
                param_threshold = 1
        elif key == ord('t'):
            param_minlen += 1
        elif key == ord('g'):
            if param_minlen > 1:
                param_minlen -= 1 
            else:
                param_minlen = 1
        elif key == ord('y'):
            param_maxgap += 1
        elif key == ord('h'):
            if param_maxgap > 1:
                param_maxgap -= 1 
            else:
                param_maxgap = 1
        elif key == ord('q'):
            break

        # print("RHO:", param_rho, "\tTHETA:", param_theta, "\tTHRESHOLD:", param_threshold)

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
