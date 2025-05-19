"""

# GIAN PAOLO D. PLARIZA
# 2022-05204

ALKANSYA - a simple Philippine Peso coin counter
CMSC 191 Final Project

This OpenCV program detects Philippine coins and
calculates the total denomination of the coins
present on frame through size and color analysis.

The application allows both image upload counting
and live feed counting

This program applies principles of:
- Python + OpenCV image handling
- image manipulation and processing
- image segmentation and contour detection
- object detection
- object tracking and motion analysis

Resources used include the following:
- Lab 4 (for reviewing contours)
- Lab 5 (for image processing)
- Lab 6 (for printing text on frame)
- Lab 8 Ball Tracking (for object tracking)
- This YouTube Video -> https://youtu.be/-iN7NDbDz3Q
Their code applies only for static camera view.
My implementation involves classifying the coin's
denomination by identifying its size based on a
reference (Coca-Cola bottle cap, red color)

"""

# importing necessary libraries
import cv2
import cvzone # only for 2x2 frame outputs
import numpy as np
import streamlit as st # GUI

# this function will detect the reference (red bottle cap)
def detectColor(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # red has two ranges in HSV
    lowerRed1 = np.array([0, 100, 100])
    upperRed1 = np.array([10, 255, 255])

    lowerRed2 = np.array([160, 100, 100])
    upperRed2 = np.array([179, 255, 255])

    # combine both masks
    mask1 = cv2.inRange(hsv, lowerRed1, upperRed1)
    mask2 = cv2.inRange(hsv, lowerRed2, upperRed2)
    mask = cv2.bitwise_or(mask1, mask2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        return max(contours, key = cv2.contourArea)
    return None

# this function preprocesses the frame
def preprocessing(frame):
    frame = cv2.GaussianBlur(frame, (5, 5), 3) # blur
    frame = cv2.Canny(frame, 110, 150) # contouring

    kernel = np.ones((3, 3), np.uint8) # defining kernel for dilation and erosion
    frame = cv2.dilate(frame, kernel, iterations = 1) # thicken lines
    frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel) # close empty gaps between lines
    frame = cv2.erode(frame, kernel, iterations = 1) # thinnen lines

    return frame

# this function processes the frame, detects coins present, classifies each coin's denomination
# and returns the frame with the processed info
def process_coins(frame, referenceSize = None):
    processedFrame = preprocessing(frame) # preprocessing the image
    coinContours, conFound = cvzone.findContours(frame, processedFrame, minArea = 200) # detects all contours with minimum area of 20
    mask = np.zeros_like(frame) # for checking on cvzone
    black = np.zeros_like(frame) # for checking on cvzone

    money = 0 # money counter printed on top left
    coinData = [] # data for 5 and 10 peso coin classification

    # detect red bottle cap
    coloredCircle = detectColor(frame)
    if coloredCircle is not None:
        referenceSize = cv2.contourArea(coloredCircle) # assign reference size to compare all coins to
        cv2.drawContours(frame, [coloredCircle], -1, (0, 255, 0), 3) # draw contour around reference

    # runs if there are any contours found
    if conFound and referenceSize:
        for contour in conFound:
            # skip overlap with reference
            if coloredCircle is not None:
                maskRef = np.zeros_like(frame[:, :, 0])
                cv2.drawContours(maskRef, [coloredCircle], -1, 255, -1)
                maskCoin = np.zeros_like(frame[:, :, 0])
                cv2.drawContours(maskCoin, [contour['cnt']], -1, 255, -1)
                overlap = cv2.bitwise_and(maskRef, maskCoin)
                if cv2.countNonZero(overlap) > 0.5 * cv2.countNonZero(maskRef):
                    continue

            # get number of sizes
            peri = cv2.arcLength(contour['cnt'], True)
            approx = cv2.approxPolyDP(contour['cnt'], 0.02 * peri, True)

            # if sides of contour is more than 5, it is a circle
            if len(approx) > 5:
                area = contour['area']
                x, y, w, h = contour['bbox']
                relativeSize = area / referenceSize # calculate its relative size
                value = 0

                # for misdetected contours
                if relativeSize < 0.45:
                    continue

                # 1 peso coin
                elif 0.45 < relativeSize < 0.65:
                    value = 1
                    money += 1

                    # printing and encircling for pretty
                    cv2.putText(frame, str(value), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                    cv2.putText(frame, str(relativeSize), (x, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                    cv2.drawContours(frame, [contour['cnt']], -1, (200, 200, 200), 2)
        
                # 5 or 10 peso coin
                elif 0.65 <= relativeSize < 0.9:
                    # classify later based on gray ratio
                    coinMask = np.zeros_like(frame[:, :, 0])
                    cv2.drawContours(coinMask, [contour['cnt']], -1, 255, -1)
                    coinROI = cv2.bitwise_and(frame, frame, mask=coinMask)
                    hsvROI = cv2.cvtColor(coinROI, cv2.COLOR_BGR2HSV)

                    # DETECT GRAY CONTENT IN COIN
                    # IF LOTS, 10 PESO
                    # IF LITTLE, 5 PESO

                    # define gray color bounds for #525252 (gray-silver)
                    lowerGray = np.array([0, 0, 50])
                    upperGray = np.array([180, 50, 130])
                    grayMask = cv2.inRange(hsvROI, lowerGray, upperGray)

                    # counting amount of gray pixels compared to total pixels inside contour
                    grayPixels = cv2.countNonZero(grayMask)
                    totalPixels = cv2.countNonZero(coinMask)
                    grayRatio = grayPixels / totalPixels if totalPixels > 0 else 0

                    coinSize = relativeSize

                    # append data to pre-initialized array for processing
                    coinData.append({
                        'contour': contour,
                        'bbox': (x, y),
                        'grayRatio': grayRatio,
                        'coinMask': coinMask,
                        'relativeSize': coinSize
                    })

                # 20 peso coin
                elif relativeSize >= 0.9:
                    value = 20
                    money += 20

                    # printing and encircling for pretty
                    cv2.putText(frame, str(value), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    cv2.putText(frame, str(relativeSize), (x, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 1)
                    cv2.drawContours(frame, [contour['cnt']], -1, (0, 165, 255), 2)

    # post-processing gray ratio based coins (5 vs 10 pesos)
    if coinData:
        grayValues = [c['grayRatio'] for c in coinData]
        grayMinimum = min(grayValues)
        grayMaximum = max(grayValues)
        # use midpoint to classify, if lower 50% = 5 pesos, if higher 50% = 10 pesos
        grayMidpoint = (grayMinimum + grayMaximum) / 2

        for c in coinData:
            x, y = c['bbox']

            # 5 pesos
            if c['grayRatio'] < grayMidpoint:
                value = 5
                money += 5
                cv2.drawContours(frame, [c['contour']['cnt']], -1, (0, 255, 255), 2)

                # printing and encircling for pretty
                cv2.putText(frame, str(value), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"{c['relativeSize']:.3f}", (x, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)
                cv2.putText(frame, f"{c['grayRatio']}", (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)

            # 10 pesos
            else:
                value = 10
                money += 10

                # printing and encircling for pretty
                cv2.drawContours(frame, [c['contour']['cnt']], -1, (200, 200, 200), 2)
                cv2.putText(frame, str(value), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                cv2.putText(frame, f"{c['relativeSize']:.3f}", (x, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                cv2.putText(frame, f"{c['grayRatio']}", (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

    cv2.putText(frame, f'php{money}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 10) # total money on top left
    return frame, money

# main function
if __name__ == '__main__':

    # streamlit interface
    money = 0
    st.title("Alkansya: A Philippine Peso Counter")

    mode = st.radio("Choose Input Mode", ("Live Camera Feed", "Upload Image")) # toggling input mode
    frame1 = st.empty()

    if mode == "Live Camera Feed":
        cap = cv2.VideoCapture(0)
        # setting auto exposure parameters bc my camera is weird
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_EXPOSURE, -6)
        stopButton = st.button("Stop Program")
        referenceSize = None

        # while not stopping program, read and process frame
        while True and not stopButton:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            processed_frame, money = process_coins(frame, referenceSize)
            frame1.image(processed_frame, channels = "BGR")

            # UNCOMMENT THIS FOR NON-GUI INTERFACE
            # stack = cvzone.stackImages([frame, processedFrame, coinContours, black], 2, 1)
            # cv2.imshow('Alkansya', stack)

            if cv2.waitKey(1) == 13:
                break

        cap.release()
        cv2.destroyAllWindows()

    elif mode == "Upload Image":
        file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if file is not None:
            # read and process uploaded image
            fileBytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            frame = cv2.imdecode(fileBytes, 1)
            frame = cv2.flip(frame, 1)
            processed_frame, money = process_coins(frame)
            frame1.image(processed_frame, channels = "BGR")

            # UNCOMMENT THIS FOR NON-GUI INTERFACE
            # stack = cvzone.stackImages([frame, processedFrame, coinContours, black], 2, 1)
            # cv2.imshow('Alkansya', stack)
