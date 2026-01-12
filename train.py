import os
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from env.unity_env import UnityLabyrinthEnv


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    config_path = "ppo_labyrinth.yaml"
    if os.path.exists(config_path):
        config = load_config(config_path)
        print("Wczytano konfigurację z pliku YAML.")
    else:
        config = {
            'learning_rate': 0.0003, 'batch_size': 64, 'n_steps': 2048,
            'gamma': 0.99, 'ent_coef': 0.01, 'total_timesteps': 100000,
            'model_save_path': "models/ppo_labyrinth_final", 'log_dir': "logs/"
        }

    os.makedirs(os.path.dirname(config['model_save_path']), exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)

    print("--- Start Serwera i Środowiska ---")

    env_maker = lambda: UnityLabyrinthEnv()

    env = DummyVecEnv([env_maker])
    env = VecFrameStack(env, n_stack=4)

    print("--- Inicjalizacja Modelu PPO (MlpPolicy) ---")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=config['learning_rate'],
        batch_size=config['batch_size'],
        n_steps=config['n_steps'],
        gamma=config['gamma'],
        ent_coef=config['ent_coef'],
        tensorboard_log=config['log_dir'],
        device="cpu"
    )

    print(f"--- Rozpoczęcie Treningu: {config['total_timesteps']} kroków ---")
    try:
        model.learn(total_timesteps=config['total_timesteps'], progress_bar=True)
        print("--- Zapisywanie Modelu ---")
        model.save(config['model_save_path'])
        print(f"Model zapisany w: {config['model_save_path']}")
    except KeyboardInterrupt:
        print("\nPrzerwano trening ręcznie. Zapisuję stan...")
        model.save(config['model_save_path'] + "_interrupted")
    finally:
        env.close()


if __name__ == "__main__":
    main()