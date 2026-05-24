import cv2
import numpy as np

try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    USE_MEDIAPIPE = True
except (ImportError, AttributeError):
    hands = None
    USE_MEDIAPIPE = False

background_frame = None
previous_centroid = None
debug_message = ""
last_hand_bbox = None
last_dimensions = (0, 0)


def get_debug_message():
    return debug_message


def get_hand_dimensions():
    return last_dimensions


def annotate_frame(frame):
    if frame is None:
        return frame

    return frame


def detect_gesture(frame):
    global background_frame, previous_centroid, debug_message, last_hand_bbox, last_dimensions

    if frame is None:
        debug_message = "No camera frame"
        last_hand_bbox = None
        last_dimensions = (0, 0)
        return "STOP"

    brightness = np.mean(frame)
    frame_std = frame.std()
    if brightness < 10:
        debug_message = f"Camera feed too dark or black (brightness={brightness:.2f})"
        return "STOP"
    if brightness > 245 or frame_std < 10:
        debug_message = f"Camera feed overexposed or blank (brightness={brightness:.2f} std={frame_std:.2f})"
        last_hand_bbox = None
        last_dimensions = (0, 0)
        return "STOP"

    if USE_MEDIAPIPE and hands is not None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            x_coords = [lm.x for lm in hand.landmark]
            y_coords = [lm.y for lm in hand.landmark]
            avg_x = np.mean(x_coords)
            avg_y = np.mean(y_coords)
            img_h, img_w = frame.shape[:2]
            x1 = int(min(x_coords) * img_w)
            x2 = int(max(x_coords) * img_w)
            y1 = int(min(y_coords) * img_h)
            y2 = int(max(y_coords) * img_h)
            last_hand_bbox = (x1, y1, x2, y2)
            last_dimensions = (x2 - x1, y2 - y1)
            debug_message = f"Hand center x={avg_x:.2f} y={avg_y:.2f}"
            if avg_x < 0.4:
                return "MOVE_LEFT"
            if avg_x > 0.6:
                return "MOVE_RIGHT"
            if avg_y < 0.4:
                return "MOVE_UP"
            if avg_y > 0.6:
                return "MOVE_DOWN"
            return "STOP"
        debug_message = "Hand not found in MediaPipe; falling back to motion detection"
        last_hand_bbox = None
        last_dimensions = (0, 0)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if background_frame is None:
        background_frame = gray.copy()
        previous_centroid = None
        debug_message = "Background frame initialized"
        return "STOP"

    frame_delta = cv2.absdiff(background_frame, gray)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        previous_centroid = None
        background_frame = cv2.addWeighted(background_frame, 0.8, gray, 0.2, 0)
        debug_message = "No motion detected"
        last_hand_bbox = None
        last_dimensions = (0, 0)
        return "STOP"

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < 2000:
        previous_centroid = None
        background_frame = cv2.addWeighted(background_frame, 0.8, gray, 0.2, 0)
        debug_message = f"Small motion area {area:.0f}"
        last_hand_bbox = None
        last_dimensions = (0, 0)
        return "STOP"

    M = cv2.moments(contour)
    if M["m00"] == 0:
        previous_centroid = None
        background_frame = cv2.addWeighted(background_frame, 0.8, gray, 0.2, 0)
        debug_message = "Invalid contour"
        last_hand_bbox = None
        last_dimensions = (0, 0)
        return "STOP"

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    if previous_centroid is None:
        previous_centroid = (cx, cy)
        background_frame = cv2.addWeighted(background_frame, 0.8, gray, 0.2, 0)
        debug_message = "Motion detected, waiting for next frame"
        x, y, w, h = cv2.boundingRect(contour)
        last_hand_bbox = (x, y, x + w, y + h)
        last_dimensions = (w, h)
        return "STOP"

    dx = cx - previous_centroid[0]
    dy = cy - previous_centroid[1]
    previous_centroid = (cx, cy)

    if abs(dx) > abs(dy):
        if dx > 15:
            direction = "MOVE_RIGHT"
        elif dx < -15:
            direction = "MOVE_LEFT"
        else:
            direction = "STOP"
    else:
        if dy > 15:
            direction = "MOVE_DOWN"
        elif dy < -15:
            direction = "MOVE_UP"
        else:
            direction = "STOP"

    x, y, w, h = cv2.boundingRect(contour)
    last_hand_bbox = (x, y, x + w, y + h)
    last_dimensions = (w, h)
    debug_message = f"Motion dx={dx} dy={dy} area={area:.0f} => {direction}"
    background_frame = cv2.addWeighted(background_frame, 0.8, gray, 0.2, 0)
    return direction
