import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from env.unity_env import UnityLabyrinthEnv


def main():
    model_path = "models/ppo_labyrinth_final"
    print(f"--- Testowanie modelu: {model_path} ---")

    env = DummyVecEnv([lambda: UnityLabyrinthEnv()])
    env = VecFrameStack(env, n_stack=4)

    try:
        model = PPO.load(model_path)
        print("Model wczytany pomyślnie!")
    except FileNotFoundError:
        print(f"Błąd: Nie znaleziono pliku {model_path}. Uruchom najpierw train.py")
        return

    obs = env.reset()
    print("Rozpoczynam pętlę sterowania...")

    while True:
        action, _ = model.predict(obs, deterministic=True)

        obs, rewards, dones, info = env.step(action)

        if dones[0]:
            print("Epizod zakończony (sukces lub limit kroków). Reset.")
            time.sleep(1.0)


if __name__ == "__main__":
    main()