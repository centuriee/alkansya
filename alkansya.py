# importing necessary libraries
import cv2
import cvzone
import numpy as np
import streamlit as st

# detect BEIGE
def detectColor(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_beige = np.array([15, 30, 100])
    upper_beige = np.array([35, 120, 255])
    
    mask = cv2.inRange(hsv, lower_beige, upper_beige)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        return max(contours, key=cv2.contourArea)
    return None

# preprocessing function
def preprocessing(frame):
    frame = cv2.GaussianBlur(frame, (5, 5), 3) # blur image
    frame = cv2.Canny(frame, 130, 180) # edge detection

    kernel = np.ones((3, 3), np.uint8)
    frame = cv2.dilate(frame, kernel, iterations = 1)
    frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel) # close any gaps
    frame = cv2.erode(frame, kernel, iterations = 1)

    return frame

cap = cv2.VideoCapture(0)

st.title("Alkansya: A Simple Philippine Peso Counter")
frame1 = st.empty()
frame2 = st.empty()
stopButton = st.button("Stop Program")

mask = None
referenceSize = None

# loops until pressing Enter key
while True and not stopButton:
    ret, frame = cap.read()

    frame = cv2.flip(frame, 1) # flips frame to reverse camera
    processedFrame = preprocessing(frame)
    coinContours, conFound = cvzone.findContours(frame, processedFrame, minArea = 20)
    mask = np.zeros_like(frame)
    black = np.zeros_like(frame)

    money = 0

    coloredCircle = detectColor(frame)
    if coloredCircle is not None:
        referenceSize = cv2.contourArea(coloredCircle)
        cv2.drawContours(frame, [coloredCircle], -1, (0,255,0), 3)

    if conFound and referenceSize:
        for count, contour in enumerate(conFound):
            peri = cv2.arcLength(contour['cnt'], True)
            approx = cv2.approxPolyDP(contour['cnt'], 0.02 * peri, True)

            # Skip if this contour is the reference circle
            if coloredCircle is not None:
                # Calculate how much the current contour overlaps with reference
                mask_ref = np.zeros_like(frame[:,:,0])
                cv2.drawContours(mask_ref, [coloredCircle], -1, 255, -1)
                
                mask_coin = np.zeros_like(frame[:,:,0])
                cv2.drawContours(mask_coin, [contour['cnt']], -1, 255, -1)
                
                overlap = cv2.bitwise_and(mask_ref, mask_coin)
                overlap_pixels = cv2.countNonZero(overlap)
                
                # Skip if more than 50% overlap with reference
                if overlap_pixels > 0.5 * cv2.countNonZero(mask_ref):
                    continue
            
            if len(approx) > 5:
                area = contour['area']
                x, y, w, h = contour['bbox']
                value = 0
                
                relativeSize = area / referenceSize

                if relativeSize < 1:
                    value = 1
                    money += 1
                elif 1 <= relativeSize < 1.3:
                    value = 5
                    money += 5
                elif relativeSize > 1.3:
                    value = 20
                    money += 20

                cv2.putText(frame, str(value), (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

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

    cv2.putText(frame, f'php{money}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    # stack = cvzone.stackImages([frame, processedFrame, coinContours, black], 2, 1)
    # cv2.imshow('Alkansya', stack)

    frame1.image(frame)

    if cv2.waitKey(1) == 13 or stopButton: # hitting Enter exits the camera capture
        break
        
# exit generated windows
cap.release()
cv2.destroyAllWindows()