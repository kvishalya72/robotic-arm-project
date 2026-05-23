import cv2
import numpy as np

from gesture_detector import annotate_frame, detect_gesture, get_debug_message
from llm_controller import llm_decision
from robot_3d_simulation import move_3d_arm, step_simulation, shutdown, get_sim_status


def open_camera():
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    for backend in backends:
        for index in (0, 1, 2):
            cap = cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                cap.release()
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)

            for _ in range(10):
                cap.read()

            ret, frame = cap.read()
            if not ret or frame is None:
                cap.release()
                continue

            mean_brightness = frame.mean()
            if mean_brightness < 5:
                cap.release()
                continue

            print(f"Camera opened on index {index} backend {backend} mean={mean_brightness:.2f}.")
            return cap
    return None


def overlay_text(frame, command, gesture, debug, sim_status):
    label_color = (255, 255, 255)
    cv2.putText(frame, f"Gesture: {gesture}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)
    cv2.putText(frame, f"Command: {command}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)
    cv2.putText(frame, f"LLM Status: {sim_status}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)
    cv2.putText(frame, f"Debug: {debug}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1)


def main():
    cap = open_camera()
    if cap is None:
        print("Error: could not open the camera on any device index.")
        print("Please make sure a webcam is connected and not in use by another app.")
        return

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error: failed to read from camera. The device may be busy or unavailable.")
            break

        gesture = detect_gesture(frame)
        command = llm_decision(gesture)
        debug = get_debug_message()

        move_3d_arm(command)
        step_simulation()

        camera_frame = cv2.resize(frame, (640, 480))
        camera_frame = annotate_frame(camera_frame)

        canvas = np.zeros((520, 1280, 3), dtype=np.uint8)
        status_lines = [
            f"Gesture: {gesture}",
            f"Command: {command}",
            f"Sim status: {get_sim_status()}",
            f"Debug: {debug}",
            "",
            "Use hand movement to control the arm:",
            " - left/right moves base rotation",
            " - up/down raises or lowers the arm",
            " - open/close gesture maps to pick/place",
        ]
        y = 40
        for line in status_lines:
            cv2.putText(canvas, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y += 35

        canvas[20:20 + 480, 640:640 + 640] = camera_frame
        cv2.rectangle(canvas, (640, 20), (1279, 499), (0, 255, 0), 2)
        cv2.putText(canvas, "Camera (right side)", (660, 515), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Robot Arm Command + Camera", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    shutdown()


if __name__ == "__main__":
    main()
