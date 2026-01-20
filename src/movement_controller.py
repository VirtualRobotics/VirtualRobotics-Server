import cv2
import numpy as np
from collections import deque
from src.config import *

diff_history = deque(maxlen=5)
oscillation_counter = 0
last_cmd_side = 0

LAST_TURN = 0
TURNING = False


def reset_state():
    global LAST_TURN, TURNING, oscillation_counter, last_cmd_side
    LAST_TURN = 0
    TURNING = False
    oscillation_counter = 0
    last_cmd_side = 0
    diff_history.clear()


def _mask_color(hsv, lower, upper):
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def decide_command_from_image(img):
    global LAST_TURN, TURNING, oscillation_counter, last_cmd_side

    h, w, _ = img.shape
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 1. LOGIKA CELU (CZERWONY)
    red1 = _mask_color(hsv, LOWER_RED1, UPPER_RED1)
    red2 = _mask_color(hsv, LOWER_RED2, UPPER_RED2)
    red_mask = cv2.bitwise_or(red1, red2)
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 50:
            M = cv2.moments(largest)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                offset_x = (cx / w) - 0.5
                TURNING = False
                oscillation_counter = 0

                if offset_x > 0.1:
                    LAST_TURN = +1
                    return TURN_RIGHT_10
                elif offset_x < -0.1:
                    LAST_TURN = -1
                    return TURN_LEFT_10
                else:
                    return MOVE_FORWARD

    # 2. LOGIKA UNIKANIA ŚCIAN (NIEBIESKI)
    blue_mask = _mask_color(hsv, LOWER_BLUE, UPPER_BLUE)

    y0 = int(h * 0.50)
    roi = blue_mask[y0:h, :]
    blue_ratio = cv2.countNonZero(roi) / float(roi.size)

    left = roi[:, :w // 2]
    right = roi[:, w // 2:]

    left_ratio = cv2.countNonZero(left) / float(left.size)
    right_ratio = cv2.countNonZero(right) / float(right.size)

    raw_diff = right_ratio - left_ratio
    diff_history.append(raw_diff)
    diff = sum(diff_history) / len(diff_history)

    center_roi = roi[:, int(w * 0.4): int(w * 0.6)]
    center_blocked = cv2.countNonZero(center_roi) / float(center_roi.size) > 0.5

    # Oscylacje
    current_side = -1 if diff > 0.1 else (1 if diff < -0.1 else 0)
    if current_side != 0 and last_cmd_side != 0 and current_side != last_cmd_side:
        oscillation_counter += 1
    else:
        oscillation_counter = max(0, oscillation_counter - 1)
    last_cmd_side = current_side

    is_dead_end = center_blocked and (left_ratio > 0.30 and right_ratio > 0.30)

    # 3. KOREKTA KIERUNKU
    if not is_dead_end:
        better_turn = 0
        if left_ratio > right_ratio + 0.05:
            better_turn = 1
        elif right_ratio > left_ratio + 0.05:
            better_turn = -1

        if better_turn != 0:
            LAST_TURN = better_turn

    if is_dead_end and LAST_TURN == 0:
        LAST_TURN = 1


    # A. Jeśli środek jest wolny -> IDŹ DO PRZODU
    if not center_blocked and abs(diff) < 0.3:
        TURNING = False
        return MOVE_FORWARD

    # B. Wyjście z pułapki (Tylko jeśli środek zablokowany i boki też)
    if is_dead_end:
        TURNING = True
        return TURN_LEFT_20 if LAST_TURN == -1 else TURN_RIGHT_20

    # C. Oscylacje (na otwartej przestrzeni)
    if oscillation_counter > 4 and not center_blocked:
        oscillation_counter = 0
        TURNING = False
        return MOVE_FORWARD

    # D. Centrowanie (tylko gdy nie ma blokady, ale jest asymetria)
    if not center_blocked:
        if blue_ratio > 0.05:
            if diff > 0.25: return TURN_LEFT_10
            if diff < -0.25: return TURN_RIGHT_10

    # E. Standardowe unikanie (gdy blokada lub narożnik)
    if center_blocked or TURNING or blue_ratio >= WALL_VERY_CLOSE_RATIO:
        TURNING = True
        return TURN_LEFT_20 if LAST_TURN == -1 else TURN_RIGHT_20

    return MOVE_FORWARD