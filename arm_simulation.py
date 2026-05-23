import cv2
import numpy as np
import pygame

pygame.init()
pygame.font.init()
font = pygame.font.SysFont(None, 24)
screen = pygame.display.set_mode((1280, 600))

base_x = 1080
base_y = 500
arm_x = 1080
arm_y = 300


def move_arm(command):
    global arm_x, arm_y
    normalized = str(command).strip().upper()
    if normalized in {"ARM_UP", "ARM_UP"}:
        arm_y = max(50, arm_y - 10)
    elif normalized in {"ARM_DOWN", "ARM_DOWN"}:
        arm_y = min(500, arm_y + 10)
    elif normalized in {"ARM_LEFT", "ROTATE_LEFT", "LEFT"}:
        arm_x = max(700, arm_x - 10)
    elif normalized in {"ARM_RIGHT", "ROTATE_RIGHT", "RIGHT"}:
        arm_x = min(1240, arm_x + 10)


def draw_arm(frame=None, gesture="", command="", debug=""):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False

    screen.fill((255, 255, 255))

    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (600, 460))
        frame = np.ascontiguousarray(frame)
        frame_surface = pygame.image.frombuffer(frame.tobytes(), (600, 460), "RGB")
        screen.blit(frame_surface, (20, 80))
        pygame.draw.rect(screen, (0, 0, 0), (15, 75, 610, 470), 3)
        label = font.render("Camera", True, (0, 0, 0))
        screen.blit(label, (20, 55))

    pygame.draw.line(screen, (0, 0, 0), (base_x, base_y), (arm_x, arm_y), 10)
    pygame.draw.circle(screen, (0, 0, 255), (arm_x, arm_y), 12)
    pygame.draw.circle(screen, (0, 0, 0), (base_x, base_y), 8)

    info_x = 660
    text_surface = font.render(f"Command: {command}", True, (0, 0, 0))
    screen.blit(text_surface, (info_x, 10))
    gesture_surface = font.render(f"Gesture: {gesture}", True, (0, 0, 0))
    screen.blit(gesture_surface, (info_x, 40))
    debug_surface = font.render(f"Debug: {debug}", True, (0, 0, 0))
    screen.blit(debug_surface, (info_x, 70))
    camera_label = font.render("Camera feed", True, (0, 0, 0))
    screen.blit(camera_label, (20, 55))

    pygame.display.update()
    return True


def shutdown():
    pygame.quit()