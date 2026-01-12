import cv2
import numpy as np

LOWER_RED1 = np.array([0, 120, 70])
UPPER_RED1 = np.array([10, 255, 255])
LOWER_RED2 = np.array([170, 120, 70])
UPPER_RED2 = np.array([180, 255, 255])

LOWER_BLUE = np.array([90, 50, 50])
UPPER_BLUE = np.array([140, 255, 255])


def get_observation_from_image(img: np.ndarray) -> np.ndarray:
    if img is None:
        return np.zeros(6, dtype=np.float32)

    h, w, _ = img.shape
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask_red1 = cv2.inRange(hsv, LOWER_RED1, UPPER_RED1)
    mask_red2 = cv2.inRange(hsv, LOWER_RED2, UPPER_RED2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    kernel = np.ones((3, 3), np.uint8)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel, iterations=2)

    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    target_visible = 0.0
    target_offset = 0.0
    target_area = 0.0

    if contours_red:
        largest = max(contours_red, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area > 50:
            target_visible = 1.0
            M = cv2.moments(largest)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                target_offset = (cx / w) - 0.5

            target_area = min(area / (w * h), 1.0)

    mask_blue = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)

    roi_y = int(h * 0.60)
    roi = mask_blue[roi_y:h, :]

    if roi.size == 0:
        return np.zeros(6, dtype=np.float32)

    roi_h, roi_w = roi.shape
    w_third = roi_w // 3

    left_zone = roi[:, :w_third]
    center_zone = roi[:, w_third:2 * w_third]
    right_zone = roi[:, 2 * w_third:]

    wall_left = cv2.countNonZero(left_zone) / left_zone.size
    wall_center = cv2.countNonZero(center_zone) / center_zone.size
    wall_right = cv2.countNonZero(right_zone) / right_zone.size

    return np.array([
        target_visible,
        target_offset,
        target_area,
        wall_left,
        wall_center,
        wall_right
    ], dtype=np.float32)