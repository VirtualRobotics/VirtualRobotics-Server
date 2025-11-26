import cv2
from config import *

def decide_command_from_image(img):
    h, w, _ = img.shape
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, LOWER_RED1, UPPER_RED1)
    mask2 = cv2.inRange(hsv, LOWER_RED2, UPPER_RED2)
    mask = mask1 | mask2

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("[PY] Brak czerwonego obiektu -> ROTATE 20 (szukam)")
        return TURN_RIGHT_20

    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)


    if area > TOO_CLOSE_THRESHOLD:
        print(f"[PY] Obiekt ZBYT BLISKO -> COFAM {area:.2f}")
        return MOVE_BACK

    if area > STOP_THRESHOLD:
        print("[PY] Blisko czerwonego obiektu -> STOP")
        return STOP

    if area < 50:
        print("[PY] Czerwony obiekt za mały -> ROTATE 20 (szukam)")
        return TURN_RIGHT_20


    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        print("[PY] Moment m00 równy zero -> ROTATE 20 (szukam)")
        return TURN_RIGHT_20

    cx = int(M["m10"] / M["m00"])

    x_norm = cx / w
    offset_x = x_norm - 0.5

    print(f"[PY] Red CX={cx}/{w}, offset={offset_x:.3f}, area={area:.1f}")

    threshold = 0.1
    if offset_x > threshold:
        print("[PY] Cel po PRAWEJ -> ROTATE 10")
        return TURN_RIGHT_10
    elif offset_x < -threshold:
        print("[PY] Cel po LEWEJ -> ROTATE -10")
        return TURN_RIGHT_10
    else:
        print("[PY] Cel wycentrowany -> MOVE 1")
        return MOVE_FORWARD
