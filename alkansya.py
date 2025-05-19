import cv2
import cvzone
import numpy as np
import streamlit as st

# Detect RED reference pin
def detectColor(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Red color spans two ranges in HSV
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])

    # Create masks for both red ranges and combine them
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        return max(contours, key=cv2.contourArea)
    return None

# Preprocess the frame
def preprocessing(frame):
    frame = cv2.GaussianBlur(frame, (5, 5), 3)
    frame = cv2.Canny(frame, 130, 180)
    kernel = np.ones((3, 3), np.uint8)
    frame = cv2.dilate(frame, kernel, iterations=1)
    frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)
    frame = cv2.erode(frame, kernel, iterations=1)
    return frame

# Streamlit interface
st.title("Alkansya: A Simple Philippine Peso Counter")

mode = st.radio("Choose Input Mode", ("Live Camera Feed", "Upload Image"))
frame1 = st.empty()
frame2 = st.empty()

if mode == "Live Camera Feed":
    cap = cv2.VideoCapture(0)
    stopButton = st.button("Stop Program")
    referenceSize = None

    while True and not stopButton:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        processedFrame = preprocessing(frame)
        coinContours, conFound = cvzone.findContours(frame, processedFrame, minArea=20)
        mask = np.zeros_like(frame)
        black = np.zeros_like(frame)

        money = 0
        coloredCircle = detectColor(frame)
        if coloredCircle is not None:
            referenceSize = cv2.contourArea(coloredCircle)
            cv2.drawContours(frame, [coloredCircle], -1, (0, 255, 0), 3)

        if conFound and referenceSize:
            for contour in conFound:
                # Skip reference overlap
                if coloredCircle is not None:
                    mask_ref = np.zeros_like(frame[:,:,0])
                    cv2.drawContours(mask_ref, [coloredCircle], -1, 255, -1)
                    mask_coin = np.zeros_like(frame[:,:,0])
                    cv2.drawContours(mask_coin, [contour['cnt']], -1, 255, -1)
                    overlap = cv2.bitwise_and(mask_ref, mask_coin)
                    if cv2.countNonZero(overlap) > 0.5 * cv2.countNonZero(mask_ref):
                        continue

                peri = cv2.arcLength(contour['cnt'], True)
                approx = cv2.approxPolyDP(contour['cnt'], 0.02 * peri, True)
                if len(approx) > 5:
                    area = contour['area']
                    x, y, w, h = contour['bbox']
                    relativeSize = area / referenceSize
                    value = 0

                    if relativeSize < 0.45:
                        pass
                    elif 0.45 < relativeSize < 0.55:
                        value = 1
                        money += 1
                    elif 0.55 <= relativeSize < 0.9:
                        value = 5
                        money += 5
                    elif relativeSize <= 0.9:
                        value = 20
                        money += 20

                    cv2.putText(frame, str(value), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    mask = np.zeros_like(frame[:,:,0])
                    cv2.drawContours(mask, [contour['cnt']], -1, 255, -1)
                    coinMask = cv2.bitwise_and(frame, frame, mask=mask)
                    coinMask = (coinMask * 0.5).astype(np.uint8)
                    hsv = cv2.cvtColor(coinMask, cv2.COLOR_BGR2HSV)
                    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255)
                    coinMask = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                    black = cv2.bitwise_or(black, coinMask)

        cv2.putText(frame, f'php{money}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        frame1.image(frame, channels="BGR")

        if cv2.waitKey(1) == 13:
            break

    cap.release()
    cv2.destroyAllWindows()

elif mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        frame = cv2.flip(frame, 1)
        processedFrame = preprocessing(frame)
        coinContours, conFound = cvzone.findContours(frame, processedFrame, minArea=20)
        mask = np.zeros_like(frame)
        black = np.zeros_like(frame)

        money = 0
        coin_data = []

        coloredCircle = detectColor(frame)
        if coloredCircle is not None:
            referenceSize = cv2.contourArea(coloredCircle)
            cv2.drawContours(frame, [coloredCircle], -1, (0, 255, 0), 3)

        if conFound and referenceSize:
            for contour in conFound:
                # Skip overlap with reference
                if coloredCircle is not None:
                    mask_ref = np.zeros_like(frame[:,:,0])
                    cv2.drawContours(mask_ref, [coloredCircle], -1, 255, -1)
                    mask_coin = np.zeros_like(frame[:,:,0])
                    cv2.drawContours(mask_coin, [contour['cnt']], -1, 255, -1)
                    overlap = cv2.bitwise_and(mask_ref, mask_coin)
                    if cv2.countNonZero(overlap) > 0.5 * cv2.countNonZero(mask_ref):
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

                    elif 0.45 < relativeSize < 0.55:
                        value = 1
                        money += 1
                        cv2.putText(frame, str(value), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        # cv2.putText(frame, str(relativeSize), (x, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 1)
            
                    elif 0.55 <= relativeSize < 0.9:
                        # Classify later based on gray ratio
                        coin_mask = np.zeros_like(frame[:,:,0])
                        cv2.drawContours(coin_mask, [contour['cnt']], -1, 255, -1)
                        coin_roi = cv2.bitwise_and(frame, frame, mask=coin_mask)
                        hsv_roi = cv2.cvtColor(coin_roi, cv2.COLOR_BGR2HSV)

                        # Define gray color bounds for #525252
                        lower_gray = np.array([0, 0, 50])
                        upper_gray = np.array([180, 50, 130])
                        gray_mask = cv2.inRange(hsv_roi, lower_gray, upper_gray)

                        gray_pixels = cv2.countNonZero(gray_mask)
                        total_pixels = cv2.countNonZero(coin_mask)
                        gray_ratio = gray_pixels / total_pixels if total_pixels > 0 else 0

                        coin_data.append({
                            'contour': contour,
                            'bbox': (x, y),
                            'gray_ratio': gray_ratio,
                            'coin_mask': coin_mask
                        })

                    elif relativeSize >= 0.9:
                        value = 20
                        money += 20
                        cv2.putText(frame, str(value), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        # cv2.putText(frame, str(relativeSize), (x, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 1)

        # Post-process gray_ratio-based coins (5 vs 10 pesos)
        if coin_data:
            gray_values = [c['gray_ratio'] for c in coin_data]
            min_gray = min(gray_values)
            max_gray = max(gray_values)
            mid_gray = (min_gray + max_gray) / 2

            for c in coin_data:
                x, y = c['bbox']
                if c['gray_ratio'] < mid_gray:
                    value = 5
                    money += 5
                else:
                    value = 10
                    money += 10

                cv2.putText(frame, str(value), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                # cv2.putText(frame, f"{c['gray_ratio']}", (x, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 1)

        cv2.putText(frame, f'php{money}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        frame1.image(frame, channels="BGR")
