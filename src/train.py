import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__)))
from env import OrbitEnv

import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback


def main():
    # Make log dir
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    # Initialize the environment — 8 parallel workers for faster, more diverse experience
    env = make_vec_env(lambda: OrbitEnv(), n_envs=8)

    # We use a separate evaluation env (single env is fine for eval)
    eval_env = make_vec_env(lambda: OrbitEnv(), n_envs=1)
    eval_callback = EvalCallback(eval_env, best_model_save_path=models_dir,
                                 log_path=log_dir, eval_freq=10000,
                                 deterministic=True, render=False)

    print("--- Initializing PPO Model ---")
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

    print("--- Starting Training (Press Ctrl+C to stop early) ---")
    try:
        model.learn(total_timesteps=2_000_000, callback=eval_callback, tb_log_name="PPO_Orbital")
    except KeyboardInterrupt:
        print("Training interrupted.")
        
    model.save(f"{models_dir}/ppo_orbit_mania")
    print("Model saved to models/ppo_orbit_mania.zip")

if __name__ == "__main__":
    main()
