import cv2
from src.config import *

# PAMIĘĆ OSTATNIEGO SKRĘTU
# -1 = lewo, +1 = prawo, 0 = brak (start)
LAST_TURN = 0
TURNING= False

def reset_state():
    global LAST_TURN, TURNING
    LAST_TURN = 0
    TURNING = False

def _mask_color(hsv, lower, upper):
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def decide_command_from_image(img):
    global LAST_TURN, TURNING

    h, w, _ = img.shape
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    red1 = _mask_color(hsv, LOWER_RED1, UPPER_RED1)
    red2 = _mask_color(hsv, LOWER_RED2, UPPER_RED2)
    red_mask = cv2.bitwise_or(red1, red2)

    contours, _ = cv2.findContours(
        red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area > 50:
            M = cv2.moments(largest)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                offset_x = (cx / w) - 0.5

                TURNING = False

                if offset_x > 0.1:
                    LAST_TURN = +1
                    return TURN_RIGHT_10
                elif offset_x < -0.1:
                    LAST_TURN = -1
                    return TURN_LEFT_10
                else:
                    return MOVE_FORWARD

    blue_mask = _mask_color(hsv, LOWER_BLUE, UPPER_BLUE)


    # bierzemy dolną część obrazu
    y0 = int(h * 0.65)
    roi = blue_mask[y0:h, :]

    blue_ratio = cv2.countNonZero(roi) / float(roi.size)

    left = roi[:, :w // 2]
    right = roi[:, w // 2:]

    # gęstość pikseli ściany po lewej i prawej
    left_ratio = cv2.countNonZero(left) / float(left.size)
    right_ratio = cv2.countNonZero(right) / float(right.size)

    diff = right_ratio - left_ratio # dodatni = więcej po prawej = skręć w lewo, ujemny = więcej po lewej = skręć w prawo

    if blue_ratio < WALL_CLOSE_RATIO and abs(diff) < (STEER_DEADZONE * 0.5): # jeśli ściana daleko i różnica mała, to nie skręcamy
        TURNING = False

    print(
        f"[PY] wall={blue_ratio:.3f} "
        f"left={left_ratio:.3f} right={right_ratio:.3f} "
        f"diff={diff:.3f} last={LAST_TURN} turning={TURNING}"
    )

    # więcej ściany po prawej → skręć w lewo
    if diff > STEER_DEADZONE:
        if LAST_TURN != -1:
            TURNING = True
        LAST_TURN = -1

    # więcej ściany po lewej → skręć w prawo
    elif diff < -STEER_DEADZONE:
        if LAST_TURN != +1:
            TURNING = True
        LAST_TURN = +1

    # jeśli brak historii skrętów, ustaw na -1 (lewo) jako domyślny
    if LAST_TURN == 0:
        LAST_TURN = -1

    # ściana daleko → idź
    if blue_ratio < WALL_CLOSE_RATIO and not TURNING:
        return MOVE_FORWARD

    if blue_ratio >= WALL_VERY_CLOSE_RATIO:
        return TURN_LEFT_20 if LAST_TURN == -1 else TURN_RIGHT_20
    else:
        return TURN_LEFT_10 if LAST_TURN == -1 else TURN_RIGHT_10
