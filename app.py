import cv2
import numpy as np
import os
import sys

from gesture_detector import annotate_frame, detect_gesture, get_debug_message
from llm_controller import llm_decision
from arm_simulation import move_arm, draw_arm, shutdown


def get_forced_camera_index():
    env_index = os.getenv("CAMERA_INDEX")
    if env_index is not None:
        try:
            return int(env_index)
        except ValueError:
            print(f"Warning: CAMERA_INDEX={env_index} is not a valid integer")
    for arg in sys.argv[1:]:
        if arg.startswith("--camera-index="):
            try:
                return int(arg.split("=", 1)[1])
            except ValueError:
                print(f"Warning: {arg} is not a valid camera index")
        elif arg.isdigit():
            return int(arg)
    return None



def open_camera():
    forced_index = get_forced_camera_index()
    indices = [forced_index] if forced_index is not None else list(range(0, 10))
    if forced_index is not None:
        print(f"Using forced camera index {forced_index}")

    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    for backend in backends:
        for index in indices:
            print(f"Trying camera index {index} with backend {backend}...")
            cap = cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                cap.release()
                print(f"  index {index} not opened")
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
                print(f"  index {index} opened but failed to read a frame")
                continue

            mean_brightness = frame.mean()
            frame_std = frame.std()
            if mean_brightness < 5:
                cap.release()
                print(f"  index {index} too dark (brightness={mean_brightness:.2f})")
                continue
            if mean_brightness > 245 or frame_std < 10:
                cap.release()
                print(f"  index {index} invalid frame: brightness={mean_brightness:.2f} std={frame_std:.2f}")
                continue

            print(f"Camera index {index} backend {backend} opened, brightness={mean_brightness:.2f}, std={frame_std:.2f}")
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