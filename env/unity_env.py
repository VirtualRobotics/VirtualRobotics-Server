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

        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(
            low=np.array([0, -0.5, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([1, 0.5, 1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )

        self.HOST = host
        self.PORT = port
        self.sock = None
        self.conn = None
        self.steps_counter = 0
        self.max_steps = 1000
        self.last_observation = np.zeros(6, dtype=np.float32)

        print(f"[ENV] Inicjalizacja środowiska na porcie {self.PORT}")
        self._start_server()

    def _start_server(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.HOST, self.PORT))
        self.sock.listen(1)
        print("Czekam na połączenie z Unity...")
        self.conn, addr = self.sock.accept()
        print(f"Połączono z Unity: {addr}")

    def _get_frame(self):
        try:
            data = b""
            header = b""
            while len(header) < 4:
                chunk = self.conn.recv(4 - len(header))
                if not chunk: return None
                header += chunk

            length = struct.unpack(">I", header)[0]

            while len(data) < length:
                chunk = self.conn.recv(length - len(data))
                if not chunk: return None
                data += chunk

            arr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            print(f"Błąd odbioru: {e}")
            return None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps_counter = 0

        try:
            self.conn.sendall(b"RESET\n")
            img = self._get_frame()
            if img is None:
                return np.zeros(6, dtype=np.float32), {}

            obs = get_observation_from_image(img)
            self.last_observation = obs
            return obs, {}
        except Exception:
            return np.zeros(6, dtype=np.float32), {}

    def step(self, action):
        self.steps_counter += 1

        command = "MOVE 0\n"
        if action == 0:
            command = "ROTATE -15\n"
        elif action == 1:
            command = "ROTATE 15\n"
        elif action == 2:
            command = "MOVE 1\n"

        try:
            self.conn.sendall(command.encode('utf-8'))
        except BrokenPipeError:
            return self.last_observation, 0, True, False, {}

        img = self._get_frame()
        if img is None:
            return self.last_observation, 0, True, False, {}

        obs = get_observation_from_image(img)
        self.last_observation = obs

        reward = -0.01

        target_visible = obs[0]
        target_offset = obs[1]
        target_area = obs[2]
        wall_left = obs[3]
        wall_center = obs[4]
        wall_right = obs[5]

        proximity_penalty = (wall_center + wall_left + wall_right) / 3.0
        if proximity_penalty > 0.1:
            reward -= proximity_penalty * 0.5

        collision_penalty = 0.0

        if action == 2:
            if wall_center > 0.15:
                collision_penalty = wall_center * 10.0
            if wall_center > 0.4:
                collision_penalty += 20.0

        elif action == 0:
            if wall_left > 0.2:
                collision_penalty = wall_left * 5.0

        elif action == 1:
            if wall_right > 0.2:
                collision_penalty = wall_right * 5.0

        reward -= collision_penalty

        if action == 2 and wall_center < 0.15:
            reward += 0.05

        if target_visible > 0.5:
            reward += 0.1
            dist_from_center = abs(target_offset)
            if dist_from_center < 0.2:
                reward += 0.2
                if action == 2 and wall_center < 0.3:
                    reward += 0.5

            if target_area > 0.05:
                reward += target_area * 2.0

            if target_area > 0.20:
                reward += 100.0
                print(f"[WIN] Area: {target_area:.3f}")
                return obs, reward, True, False, {}
        else:
            if action != 2:
                reward -= 0.02

        truncated = False
        if self.steps_counter >= self.max_steps:
            truncated = True

        return obs, reward, False, truncated, {}

    def close(self):
        if self.conn: self.conn.close()
        if self.sock: self.sock.close()