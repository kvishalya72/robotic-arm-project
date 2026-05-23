import cv2
import numpy as np

from gesture_detector import annotate_frame, detect_gesture, get_debug_message
from llm_controller import llm_decision
from arm_simulation import move_arm, draw_arm, shutdown


def open_camera():
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    for backend in backends:
        for index in range(0, 6):
            cap = cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                cap.release()
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)

            # Warm up the camera so it can expose correctly
            for _ in range(10):
                cap.read()

            ret, frame = cap.read()
            if not ret or frame is None:
                cap.release()
                continue

            mean_brightness = frame.mean()
            print(f"Camera index {index} backend {backend} opened, brightness={mean_brightness:.2f}")
            return cap
    return None


def main():
    cap = open_camera()
    if cap is None:
        print("Error: could not open the camera on any device index.")
        print("If you have multiple cameras, try changing the index in app.py.")
        return

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error: failed to read from camera. The device may be busy or unavailable.")
            break

        # Brighten low-light frames for display while keeping detection on the original feed.
        display_frame = cv2.convertScaleAbs(frame, alpha=1.4, beta=30)
        gesture = detect_gesture(frame)
        command = llm_decision(gesture)
        debug = get_debug_message()

        annotated_frame = annotate_frame(display_frame.copy())
        move_arm(command)
        if not draw_arm(annotated_frame, gesture, command, debug):
            break

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    shutdown()


if __name__ == "__main__":
    main()