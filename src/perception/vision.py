import cv2
import numpy as np

# Zakresy kolorów dla czerwonego (dostosuj jeśli w Unity oświetlenie zmienia kolor)
LOWER_RED1 = np.array([0, 100, 80])
UPPER_RED1 = np.array([10, 255, 255])
LOWER_RED2 = np.array([170, 100, 80])
UPPER_RED2 = np.array([180, 255, 255])


def get_observation_from_image(img: np.ndarray) -> np.ndarray:
    """
    Przetwarza obraz z Unity i zwraca wektor obserwacji dla RL.

    Args:
        img: Obraz w formacie BGR (bezpośrednio z cv2.imdecode).

    Returns:
        np.array: Tablica 3 liczb float32:
                  1. Czy widzę cel? (1.0 = tak, 0.0 = nie)
                  2. Gdzie jest cel? (-0.5 = lewo, 0.0 = środek, 0.5 = prawo)
                  3. Jak blisko? (0.0 = daleko, 1.0 = bardzo blisko)
    """
    if img is None:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    h, w, _ = img.shape
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Maska na kolor czerwony
    mask1 = cv2.inRange(hsv, LOWER_RED1, UPPER_RED1)
    mask2 = cv2.inRange(hsv, LOWER_RED2, UPPER_RED2)
    mask = mask1 | mask2

    # Redukcja szumów
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- BRAK CELU (ŚCIANA) ---
    if not contours:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)

    # --- SZUM (ZA MAŁE) ---
    if area < 50:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    # --- WIDZĘ CEL ---
    cx = int(M["m10"] / M["m00"])

    # Normalizacja pozycji X
    offset_x = (cx / w) - 0.5

    # Normalizacja wielkości (przybliżenie odległości)
    norm_area = min(area / (w * h), 1.0)

    return np.array([1.0, offset_x, norm_area], dtype=np.float32)