import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray_img, (5, 5), 0)
    edgeDetectedImage = cv2.Canny(blur, 25, 75)
    return edgeDetectedImage

def roi(img):
    height = img.shape[0]
    triangle = np.array([[(250,height), (1050, height), (600, 250)]])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, triangle, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def disaplyLines(img, lines):
    line_img = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_img, (x1,y1), (x2,y2), (255, 0, 0), 10)
    return line_img

def get_coords(img, line):
    slope, intercept = line
    y1 = img.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2 ])

def average_slope_intercept(img, lines):
    left = []
    right = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        params = np.polyfit((x1, x2), (y1, y2), 1)
        slope = params[0]
        intercept = params[1]
        if slope < 0:
            left.append((slope, intercept))
        else:
            right.append((slope, intercept))
    
    left_avg = np.average(left, axis=0)
    right_avg = np.average(right, axis=0)
    left_line = get_coords(img, left_avg)
    right_line = get_coords(img, right_avg)
    return np.array([left_line, right_line])


video = cv2.VideoCapture("test2.mp4")
while(video.isOpened()):
    i, frame = video.read() 
    img_canny = canny(frame)
    interested_region = roi(img_canny)
    lines = cv2.HoughLinesP(interested_region, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines=average_slope_intercept(frame, lines)
    img_with_lines = disaplyLines(frame, averaged_lines)
    final_img = cv2.addWeighted(frame, 0.8, img_with_lines, 1, 1)
    cv2.imshow('result', final_img )
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()


