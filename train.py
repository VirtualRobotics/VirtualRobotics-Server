import os
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from enviroment.unity_env import UnityLabyrinthEnv


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    config_path = "configs/ppo_labyrinth.yaml"
    if os.path.exists(config_path):
        config = load_config(config_path)
    else:
        # Domyślne wartości
        config = {
            'learning_rate': 0.0003, 'batch_size': 64, 'n_steps': 2048,
            'gamma': 0.99, 'ent_coef': 0.01, 'total_timesteps': 50000,
            'model_save_path': "models/ppo_labyrinth", 'log_dir': "logs/"
        }

    os.makedirs(os.path.dirname(config['model_save_path']), exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)

    print("--- Inicjalizacja Środowiska ---")
    env = UnityLabyrinthEnv()

    # Owijanie środowiska w Vectory i FrameStack (Pamięć)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)

    print("--- Konfiguracja Agenta ---")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=config['learning_rate'],
        batch_size=config['batch_size'],
        n_steps=config['n_steps'],
        gamma=config['gamma'],
        ent_coef=config.get('ent_coef', 0.0),
        tensorboard_log=config['log_dir'],
        device="cpu"  # Wymuszamy CPU dla stabilności
    )

    print(f"--- START TRENINGU ({config['total_timesteps']} kroków) ---")
    try:
        # progress_bar=False rozwiązuje problem z brakującymi bibliotekami
        model.learn(total_timesteps=config['total_timesteps'], progress_bar=False)
    except KeyboardInterrupt:
        print("\n[STOP] Przerwano ręcznie. Zapisuję model...")

    model.save(config['model_save_path'])
    print(f"[SUKCES] Model zapisany w: {config['model_save_path']}")


if __name__ == "__main__":
    main()