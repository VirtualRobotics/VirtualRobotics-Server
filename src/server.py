import struct
import socket
import cv2
import numpy as np
from typing import Optional

from src.config import Config
from src.movement_controller import MovementController


def receive_exact(connection: socket.socket, length: int) -> Optional[bytes]:
    data = b""
    while len(data) < length:
        packet = connection.recv(length - len(data))
        if not packet:
            return None
        data += packet
    return data


def handle_client(connection: socket.socket, address, debug: bool = False):
    print(f"[PY] Connected to {address}")

    navigator = MovementController(Config)
    navigator.reset_state()

    try:
        while True:
            # 1. Read Header (4 bytes = uint32 payload length)
            header = receive_exact(connection, 4)
            if not header:
                print("[PY] Client disconnected (missing header)")
                break

            (length,) = struct.unpack(">I", header)

            # 2. Read Image Data
            frame_bytes = receive_exact(connection, length)
            if not frame_bytes:
                print("[PY] Client disconnected (incomplete frame)")
                break

            np_data = np.frombuffer(frame_bytes, np.uint8)
            img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

            if debug and img is not None:
                cv2.imshow(f"Agent View {address}", img)
                cv2.waitKey(1)

            if img is None:
                print("[PY] Error: Failed to decode image")
                # Fallback action if camera fails
                response = Config.Cmd.TURN_RIGHT_10
            else:
                response = navigator.decide_command(img)

            connection.sendall(response.encode("utf-8"))
            print(f"[PY] Sent: {response.strip()}")

    except ConnectionError:
        print("[PY] Connection error")
    except Exception as e:
        print(f"[PY] Unexpected error: {e}")
    finally:
        if debug:
            cv2.destroyAllWindows()
        print(f"[PY] Session ended for {address}")
