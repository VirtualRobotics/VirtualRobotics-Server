import socket
import cv2
import numpy as np
import struct

HOST = "127.0.0.1"
PORT = 5000

IMAGE_PATH = r"C:/Users/oskar/AppData/LocalLow/DefaultCompany/VirtualRobotics\agent_frame.png"

def receive_frame(connection, length):
    data = b""
    while len(data) < length:
        packet = connection.recv(length - len(data))
        if not packet:
            raise ConnectionError("Połączenie zamknięte przez klienta")
        data += packet
    return data


def decide_command_from_image(img):
    look_message = "ROTATE 20\n"
    move_message = "MOVE 1\n"
    turn_left_message = "ROTATE -10\n"
    turn_right_message = "ROTATE 10\n"

    h, w, _ = img.shape
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 100, 80])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 80])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 | mask2

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("[PY] Brak czerwonego obiektu -> ROTATE 20 (szukam)")
        return look_message

    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)

    if area < 50:
        print("[PY] Czerwony obiekt za mały -> ROTATE 20 (szukam)")
        return look_message

    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        print("[PY] Moment m00 równy zero -> ROTATE 20 (szukam)")
        return look_message

    cx = int(M["m10"] / M["m00"])

    x_norm = cx / w
    offset_x = x_norm - 0.5

    print(f"[PY] Red CX={cx}/{w}, offset={offset_x:.3f}, area={area:.1f}")

    threshold = 0.1
    if offset_x > threshold:
        print("[PY] Cel po PRAWEJ -> ROTATE 10")
        return turn_right_message
    elif offset_x < -threshold:
        print("[PY] Cel po LEWEJ -> ROTATE -10")
        return turn_left_message
    else:
        print("[PY] Cel wycentrowany -> MOVE 1")
        return move_message


def handle_client(connection, address, debug=False):
    print(f"[PY] Połączono z {address}")
    with connection:
        while True:
            try:
                header = receive_frame(connection, 4)
                (length,) = struct.unpack(">I", header)

                frame_bytes = receive_frame(connection, length)
                np_data = np.frombuffer(frame_bytes, np.uint8)
                img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

                if debug:
                    cv2.imshow("Agent view", img)
                    cv2.waitKey(1)

                if img is None:
                    print("[PY] Nie udało się zdekodować obrazu")
                    response = "ROTATE 20\n"
                else:
                    response = decide_command_from_image(img)

                connection.sendall(response.encode('utf-8'))
                print(f"[PY] Wysłano do Unity: {response.strip()}")

            except ConnectionError:
                print("[PY] Klient się rozłączył")
                break


def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen(1)
        print(f"[PY] Serwer nasłuchuje na {HOST}:{PORT}")
        while True:
            connection, address = s.accept()
            handle_client(connection, address, debug=True)


if __name__ == "__main__":
    main()
