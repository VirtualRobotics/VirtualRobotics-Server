import socket
import struct
import numpy as np
import cv2
import os
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# ================= KONFIGURACJA =================
HOST = "127.0.0.1"
PORT = 5000
MODEL_DIR = "models_safe"
LOG_DIR = "logs_safe"
IMG_SIZE = 84

# Jeśli czerwony zajmie 15% ekranu -> Zwycięstwo
WIN_AREA_THRESHOLD = 0.15
MAX_STEPS = 600

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


class UnitySafeEnv(gym.Env):
    def __init__(self):
        super(UnitySafeEnv, self).__init__()

        # 0=Lewo, 1=Prawo, 2=Prosto
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8
        )

        self.server_socket = None
        self.connection = None
        self.steps_counter = 0

        # PAMIĘĆ: Czy w ostatniej klatce widziałem cel
        # Na starcie zakładamy False, żeby nie ruszył na ślepo.
        self.last_target_found = False

        print("[INIT] Startuję serwer...")
        self._start_server()

    def _start_server(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((HOST, PORT))
        self.server_socket.listen(1)
        print(f"[SERVER] Czekam na Unity na {HOST}:{PORT}...")
        self.connection, addr = self.server_socket.accept()
        self.connection.settimeout(60.0)
        print(f"[SERVER] Połączono z: {addr}")

    def _receive_frame(self):
        try:
            header = b""
            while len(header) < 4:
                chunk = self.connection.recv(4 - len(header))
                if not chunk: raise ConnectionError("Brak nagłówka")
                header += chunk
            (length,) = struct.unpack(">I", header)
            data = b""
            while len(data) < length:
                chunk = self.connection.recv(length - len(data))
                if not chunk: raise ConnectionError("Urwany obraz")
                data += chunk
            return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"[ERROR] {e}")
            self.close()
            raise e

    def _analyze_image(self, img):
        """Standardowa detekcja czerwonego"""
        h, w, _ = img.shape
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Zakresy kolorów
        mask1 = cv2.inRange(hsv, np.array([0, 100, 80]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([170, 100, 80]), np.array([180, 255, 255]))
        mask = mask1 | mask2

        # Czyszczenie
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return False, 0.0, 0.0, mask

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area < 50: return False, 0.0, 0.0, mask

        M = cv2.moments(largest)
        if M["m00"] == 0: return False, 0.0, 0.0, mask

        cx = int(M["m10"] / M["m00"])
        offset_x = (cx / w) - 0.5
        norm_area = area / (w * h)

        return True, offset_x, norm_area, mask

    def step(self, action):
        self.steps_counter += 1


        override_penalty = 0.0

        # Logika: Jeśli chcesz jechać (Action 2), ALE w ostatniej klatce
        # nie widziałeś celu (last_target_found == False), to ZABRANIAM.

        final_command = ""

        if action == 2 and not self.last_target_found:
            # SYTUACJA: Agent chce jechać na ślepo.
            # REAKCJA: Wymuszamy obrót w lewo (szukanie) zamiast ruchu.
            final_command = "ROTATE -20\n"

            # Dajemy karę, żeby sieć wiedziała, że podjęła złą decyzję
            override_penalty = -1.0
            # print(" [SAFETY] Zablokowano ruch")
        else:
            # Normalne sterowanie
            if action == 0:
                final_command = "ROTATE -10\n"
            elif action == 1:
                final_command = "ROTATE 10\n"
            elif action == 2:
                final_command = "MOVE 1\n"

        # Wysłanie komendy do Unity
        try:
            self.connection.sendall(final_command.encode('utf-8'))
        except (BrokenPipeError, OSError):
            return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8), 0, True, False, {}

        # ==================================================================

        # Pobranie nowej klatki
        full_img = self._receive_frame()
        found, offset_x, norm_area, mask = self._analyze_image(full_img)

        # Aktualizujemy pamięć dla następnego kroku
        self.last_target_found = found

        # --- OBLICZANIE NAGRODY ---
        reward = 0.0 + override_penalty
        done = False

        reward -= 0.01  # Kara za czas

        if found:
            # Widzę cel!
            if abs(offset_x) < 0.15:  # Jest w miarę na środku
                if action == 2:
                    # Idealnie: Widzę na środku i jadę
                    reward += 1.0
                    reward += norm_area * 10
                else:
                    # Widzę na środku, a kręcę się? Błąd.
                    reward -= 0.5
            else:
                # Widzę, ale z boku (wymaga celowania)
                if action == 2:
                    reward -= 0.5  # Nie jedź jak nie wycelowałeś
                else:
                    # Nagroda za obrót w dobrą stronę
                    if (offset_x < 0 and action == 0) or (offset_x > 0 and action == 1):
                        reward += 0.5
                    else:
                        reward -= 0.2

            # Warunek wygranej (dojechanie blisko)
            if norm_area >= WIN_AREA_THRESHOLD:
                reward += 20.0
                done = True
                print(f"[WIN] Dojechałem! Area: {norm_area:.3f}")
        else:
            # Nie widzę celu
            if action == 2:
                # musi dostać karę za samą chęć jazdy na ślepo.
                reward -= 1.0
            else:
                # Szukanie (obrót) jest dobre, gdy nic nie widzisz
                reward += 0.1

        truncated = (self.steps_counter >= MAX_STEPS)
        obs = cv2.resize(full_img, (IMG_SIZE, IMG_SIZE))

        # Debug
        cv2.imshow("Brain View", obs)
        cv2.imshow("Brain Mask", mask)
        cv2.waitKey(1)

        return obs, reward, done, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps_counter = 0
        self.last_target_found = False  # Reset pamięci po restarcie
        try:
            self.connection.sendall(b"RESET\n")
            img = self._receive_frame()
            obs = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            # Sprawdzamy, czy na starcie coś widać
            found, _, _, _ = self._analyze_image(img)
            self.last_target_found = found
            return obs, {}
        except Exception:
            return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8), {}

    def close(self):
        if self.connection: self.connection.close()
        if self.server_socket: self.server_socket.close()
        cv2.destroyAllWindows()


# ================= START =================
if __name__ == "__main__":
    env = UnitySafeEnv()
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)

    model = PPO("CnnPolicy", env, verbose=1, learning_rate=0.0003, batch_size=64)

    print("--- START (TRYB BEZPIECZNY) ---")
    print("Agent fizycznie NIE MOŻE jechać, jeśli nie widział celu w poprzedniej klatce.")

    try:
        model.learn(total_timesteps=100000, progress_bar=True)
        model.save("rl_safe_final")
    finally:
        env.close()