import socket
import struct
import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces
from src.perception.vision import get_observation_from_image


class UnityLabyrinthEnv(gym.Env):
    def __init__(self, host='127.0.0.1', port=5000):
        super(UnityLabyrinthEnv, self).__init__()

        # AKCJE: 0=Lewo, 1=Prawo, 2=Prosto
        self.action_space = spaces.Discrete(3)

        # OBSERWACJA: [czy_widzi, offset_x, area]
        self.observation_space = spaces.Box(
            low=np.array([0, -0.5, 0], dtype=np.float32),
            high=np.array([1, 0.5, 1], dtype=np.float32),
            dtype=np.float32
        )

        self.HOST = host
        self.PORT = port
        self.sock = None
        self.conn = None

        # Pamięć ostatniej obserwacji
        self.last_observation = np.array([0, 0, 0], dtype=np.float32)

        print(f"[ENV] Inicjalizacja środowiska na porcie {self.PORT}")
        self._start_server()

    def _start_server(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.HOST, self.PORT))
        self.sock.listen(1)
        print(f"[SERVER] Czekam na połączenie z Unity...")
        self.conn, addr = self.sock.accept()
        print(f"[SERVER] Połączono z: {addr}")

    def _send_command(self, cmd: str):
        try:
            self.conn.sendall(cmd.encode('utf-8'))
        except (BrokenPipeError, OSError):
            print("[SERVER] Unity rozłączone!")
            exit()

    def _get_frame(self):
        try:
            header = b""
            while len(header) < 4:
                chunk = self.conn.recv(4 - len(header))
                if not chunk: return None
                header += chunk
            (length,) = struct.unpack(">I", header)

            data = b""
            while len(data) < length:
                chunk = self.conn.recv(length - len(data))
                if not chunk: return None
                data += chunk

            np_img = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            print(f"[ERROR] Błąd odbioru ramki: {e}")
            return None

    def step(self, action):
        command = "MOVE 0\n"
        if action == 2:  # Prosto
            command = "MOVE 1\n"
        elif action == 0:  # Lewo
            command = "ROTATE -10\n"
        elif action == 1:  # Prawo
            command = "ROTATE 10\n"

        # 1. Wyślij ruch
        self._send_command(command)

        # 2. Odbierz nowy obraz
        img = self._get_frame()
        if img is None:
            return self.last_observation, 0, True, False, {}

        # 3. Przetwórz obraz
        obs = get_observation_from_image(img)
        self.last_observation = obs

        # --- SYSTEM NAGRÓD (Zoptymalizowany pod labirynt) ---
        reward = -0.005  # Mała kara za czas

        # Sytuacja A: Agent widzi cel
        if obs[0] > 0.5:
            reward += 0.1
            reward += (0.5 - abs(obs[1])) * 2.0

            if action == 2:
                reward += 0.5

                # WARUNEK ZWYCIĘSTWA
            if obs[2] > 0.15:
                reward += 20.0
                print(f"[WIN] Cel osiągnięty! Wielkość: {obs[2]:.2f}")
                return obs, reward, True, False, {}

        # Sytuacja B: Agent nie widzi celu
        else:
            # Zachęta do eksploracji w ciemno
            if action == 2:
                reward += 0.05
            # Kara za kręcenie się bez celu
            elif action == 0 or action == 1:
                reward -= 0.02

        return obs, reward, False, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._send_command("RESET\n")

        img = self._get_frame()
        if img is not None:
            self.last_observation = get_observation_from_image(img)
        else:
            self.last_observation = np.array([0, 0, 0], dtype=np.float32)

        return self.last_observation, {}