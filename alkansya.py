import cv2
import cvzone
import numpy as np
import streamlit as st

# Detect red reference
def detectColor(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Red color spans two ranges in HSV
    lowerRed1 = np.array([0, 100, 100])
    upperRed1 = np.array([10, 255, 255])

    lowerRed2 = np.array([160, 100, 100])
    upperRed2 = np.array([179, 255, 255])

    # Create masks for both red ranges and combine them
    mask1 = cv2.inRange(hsv, lowerRed1, upperRed1)
    mask2 = cv2.inRange(hsv, lowerRed2, upperRed2)
    mask = cv2.bitwise_or(mask1, mask2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        return max(contours, key = cv2.contourArea)
    return None

# Preprocess the frame
def preprocessing(frame):
    frame = cv2.GaussianBlur(frame, (5, 5), 3)
    frame = cv2.Canny(frame, 110, 150)

    kernel = np.ones((3, 3), np.uint8)
    frame = cv2.dilate(frame, kernel, iterations=1)
    frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)
    frame = cv2.erode(frame, kernel, iterations=1)

    return frame

# Streamlit interface
st.title("Alkansya: A Philippine Peso Counter")

mode = st.radio("Choose Input Mode", ("Live Camera Feed", "Upload Image"))
frame1 = st.empty()

if mode == "Live Camera Feed":
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 for manual mode on many systems
    cap.set(cv2.CAP_PROP_EXPOSURE, -6)  # Set exposure (smaller value = brighter; camera dependent)
    stopButton = st.button("Stop Program")
    referenceSize = None

    while True and not stopButton:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        processedFrame = preprocessing(frame)
        coinContours, conFound = cvzone.findContours(frame, processedFrame, minArea = 20)
        mask = np.zeros_like(frame)
        black = np.zeros_like(frame)

        money = 0
        coinData = []

        coloredCircle = detectColor(frame)
        if coloredCircle is not None:
            referenceSize = cv2.contourArea(coloredCircle)
            cv2.drawContours(frame, [coloredCircle], -1, (0, 255, 0), 3)

        if conFound and referenceSize:
            for contour in conFound:
                # Skip overlap with reference
                if coloredCircle is not None:
                    maskRef = np.zeros_like(frame[:, :, 0])
                    cv2.drawContours(maskRef, [coloredCircle], -1, 255, -1)
                    maskCoin = np.zeros_like(frame[:, :, 0])
                    cv2.drawContours(maskCoin, [contour['cnt']], -1, 255, -1)
                    overlap = cv2.bitwise_and(maskRef, maskCoin)
                    if cv2.countNonZero(overlap) > 0.5 * cv2.countNonZero(maskRef):
                        continue

                peri = cv2.arcLength(contour['cnt'], True)
                approx = cv2.approxPolyDP(contour['cnt'], 0.02 * peri, True)
                if len(approx) > 5:
                    area = contour['area']
                    x, y, w, h = contour['bbox']
                    relativeSize = area / referenceSize
                    value = 0

                    if relativeSize < 0.45:
                        continue

                    elif 0.45 < relativeSize < 0.65:
                        value = 1
                        money += 1
                        cv2.putText(frame, str(value), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                        cv2.putText(frame, str(relativeSize), (x, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                        cv2.drawContours(frame, [contour['cnt']], -1, (200, 200, 200), 2)
            
                    elif 0.65 <= relativeSize < 0.9:
                        # Classify later based on gray ratio
                        coinMask = np.zeros_like(frame[:, :, 0])
                        cv2.drawContours(coinMask, [contour['cnt']], -1, 255, -1)
                        coinROI = cv2.bitwise_and(frame, frame, mask = coinMask)
                        hsvROI = cv2.cvtColor(coinROI, cv2.COLOR_BGR2HSV)

                        # Define gray color bounds for #525252
                        lowerGray = np.array([0, 0, 50])
                        upperGray = np.array([180, 50, 130])
                        grayMask = cv2.inRange(hsvROI, lowerGray, upperGray)

                        grayPixels = cv2.countNonZero(grayMask)
                        totalPixels = cv2.countNonZero(coinMask)
                        grayRatio = grayPixels / totalPixels if totalPixels > 0 else 0

                        coinSize = relativeSize

                        coinData.append({
                            'contour': contour,
                            'bbox': (x, y),
                            'grayRatio': grayRatio,
                            'coinMask': coinMask,
                            'relativeSize': coinSize
                        })

                    elif relativeSize >= 0.9:
                        value = 20
                        money += 20
                        cv2.putText(frame, str(value), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                        cv2.putText(frame, str(relativeSize), (x, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 1)
                        cv2.drawContours(frame, [contour['cnt']], -1, (0, 165, 255), 2)

        # Post-process grayRatio-based coins (5 vs 10 pesos)
        if coinData:
            grayValues = [c['grayRatio'] for c in coinData]
            grayMinimum = min(grayValues)
            grayMaximum = max(grayValues)
            grayMidpoint = (grayMinimum + grayMaximum) / 2

            for c in coinData:
                x, y = c['bbox']
                if c['grayRatio'] < grayMidpoint:
                    value = 5
                    money += 5
                    cv2.drawContours(frame, [c['contour']['cnt']], -1, (0, 255, 255), 2)
                    cv2.putText(frame, str(value), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(frame, f"{c['relativeSize']:.3f}", (x, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)
                    cv2.putText(frame, f"{c['grayRatio']}", (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)
                else:
                    value = 10
                    money += 10
                    cv2.drawContours(frame, [c['contour']['cnt']], -1, (200, 200, 200), 2)
                    cv2.putText(frame, str(value), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                    cv2.putText(frame, f"{c['relativeSize']:.3f}", (x, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                    cv2.putText(frame, f"{c['grayRatio']}", (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

        cv2.putText(frame, f'php{money}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 10)
        frame1.image(frame, channels = "BGR")

        # stack = cvzone.stackImages([frame, processedFrame, coinContours, black], 2, 1)
        # cv2.imshow('Alkansya', stack)

        if cv2.waitKey(1) == 13:
            break

    cap.release()
    cv2.destroyAllWindows()

elif mode == "Upload Image":
    file = st.file_uploader("Upload an image", type= ["jpg", "jpeg", "png"])
    if file is not None:
        fileBytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        frame = cv2.imdecode(fileBytes, 1)
        frame = cv2.flip(frame, 1)
        processedFrame = preprocessing(frame)
        coinContours, conFound = cvzone.findContours(frame, processedFrame, minArea=20)
        mask = np.zeros_like(frame)
        black = np.zeros_like(frame)

        money = 0
        coinData = []

        coloredCircle = detectColor(frame)
        if coloredCircle is not None:
            referenceSize = cv2.contourArea(coloredCircle)
            cv2.drawContours(frame, [coloredCircle], -1, (0, 255, 0), 3)

        if conFound and referenceSize:
            for contour in conFound:
                # Skip overlap with reference
                if coloredCircle is not None:
                    maskRef = np.zeros_like(frame[:, :, 0])
                    cv2.drawContours(maskRef, [coloredCircle], -1, 255, -1)
                    maskCoin = np.zeros_like(frame[:, :, 0])
                    cv2.drawContours(maskCoin, [contour['cnt']], -1, 255, -1)
                    overlap = cv2.bitwise_and(maskRef, maskCoin)
                    if cv2.countNonZero(overlap) > 0.5 * cv2.countNonZero(maskRef):
                        continue

                peri = cv2.arcLength(contour['cnt'], True)
                approx = cv2.approxPolyDP(contour['cnt'], 0.02 * peri, True)
                if len(approx) > 5:
                    area = contour['area']
                    x, y, w, h = contour['bbox']
                    relativeSize = area / referenceSize
                    value = 0

                    if relativeSize < 0.45:
                        continue

                    elif 0.45 < relativeSize < 0.65:
                        value = 1
                        money += 1
                        cv2.putText(frame, str(value), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                        cv2.putText(frame, str(relativeSize), (x, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                        cv2.drawContours(frame, [contour['cnt']], -1, (200, 200, 200), 2)
            
                    elif 0.65 <= relativeSize < 0.9:
                        # Classify later based on gray ratio
                        coinMask = np.zeros_like(frame[:, :, 0])
                        cv2.drawContours(coinMask, [contour['cnt']], -1, 255, -1)
                        coinROI = cv2.bitwise_and(frame, frame, mask=coinMask)
                        hsvROI = cv2.cvtColor(coinROI, cv2.COLOR_BGR2HSV)

                        # Define gray color bounds for #525252
                        lowerGray = np.array([0, 0, 50])
                        upperGray = np.array([180, 50, 130])
                        grayMask = cv2.inRange(hsvROI, lowerGray, upperGray)

                        grayPixels = cv2.countNonZero(grayMask)
                        totalPixels = cv2.countNonZero(coinMask)
                        grayRatio = grayPixels / totalPixels if totalPixels > 0 else 0

                        coinSize = relativeSize

                        coinData.append({
                            'contour': contour,
                            'bbox': (x, y),
                            'grayRatio': grayRatio,
                            'coinMask': coinMask,
                            'relativeSize': coinSize
                        })

                    elif relativeSize >= 0.9:
                        value = 20
                        money += 20
                        cv2.putText(frame, str(value), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                        cv2.putText(frame, str(relativeSize), (x, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 1)
                        cv2.drawContours(frame, [contour['cnt']], -1, (0, 165, 255), 2)

        # Post-process grayRatio-based coins (5 vs 10 pesos)
        if coinData:
            grayValues = [c['grayRatio'] for c in coinData]
            grayMinimum = min(grayValues)
            grayMaximum = max(grayValues)
            grayMidpoint = (grayMinimum + grayMaximum) / 2

            for c in coinData:
                x, y = c['bbox']
                if c['grayRatio'] < grayMidpoint:
                    value = 5
                    money += 5
                    cv2.drawContours(frame, [c['contour']['cnt']], -1, (0, 255, 255), 2)
                    cv2.putText(frame, str(value), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(frame, f"{c['relativeSize']:.3f}", (x, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)
                    cv2.putText(frame, f"{c['grayRatio']}", (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)
                else:
                    value = 10
                    money += 10
                    cv2.drawContours(frame, [c['contour']['cnt']], -1, (200, 200, 200), 2)
                    cv2.putText(frame, str(value), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                    cv2.putText(frame, f"{c['relativeSize']:.3f}", (x, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                    cv2.putText(frame, f"{c['grayRatio']}", (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

        cv2.putText(frame, f'php{money}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 10)
        frame1.image(frame, channels = "BGR")