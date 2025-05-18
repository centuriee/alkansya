# importing necessary libraries
import cv2
import cvzone
import numpy as np

def preprocessing(frame):
    frame = cv2.GaussianBlur(frame, (5, 5), 3) # blur image
    frame = cv2.Canny(frame, 130, 180) # edge detection

    kernel = np.ones((3, 3), np.uint8)
    frame = cv2.dilate(frame, kernel, iterations = 1)
    frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel) # close any gaps
    frame = cv2.erode(frame, kernel, iterations = 1)

    return frame

cap = cv2.VideoCapture(0)
mask = None

# loops until pressing Enter key
while True:
    ret, frame = cap.read()

    frame = cv2.flip(frame, 1) # flips frame to reverse camera
    processedFrame = preprocessing(frame)
    coinContours, conFound = cvzone.findContours(frame, processedFrame, minArea = 20)
    mask = np.zeros_like(frame)
    black = np.zeros_like(frame)

    if conFound:
        for contour in conFound:
            peri = cv2.arcLength(contour['cnt'], True)
            approx = cv2.approxPolyDP(contour['cnt'], 0.02 * peri, True)
            
            if len(approx) > 5:
                mask = np.zeros_like(frame[:,:,0])
            
                # Draw filled white contour on mask
                cv2.drawContours(mask, [contour['cnt']], -1, 255, -1)
                
                # Apply mask to original image and add to black background
                coinMask = cv2.bitwise_and(frame, frame, mask=mask)
                
                # lower brightness
                coinMask = (coinMask * 0.5).astype(np.uint8)

                # increase saturation
                hsv = cv2.cvtColor(coinMask, cv2.COLOR_BGR2HSV)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255)  # boost by 1.5
                coinMask = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

                # bitwise or black and coinmask
                black = cv2.bitwise_or(black, coinMask)

    stack = cvzone.stackImages([frame, processedFrame, coinContours, black], 2, 1)
    cv2.imshow('Alkansya', stack)
    
    if cv2.waitKey(1) == 13: # hitting Enter exits the camera capture
        break
        
# exit generated windows
cap.release()
cv2.destroyAllWindows()