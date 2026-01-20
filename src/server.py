import struct
import src.movement_controller as mc
from src.movement_controller import *

def receive_frame(connection, length):
    data = b""
    while len(data) < length:
        packet = connection.recv(length - len(data))
        if not packet:
            raise ConnectionError("Połączenie zamknięte przez klienta")
        data += packet
    return data

def handle_client(connection, address, debug=False):
    print(f"[PY] Połączono z {address}")

    mc.reset_state()

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
