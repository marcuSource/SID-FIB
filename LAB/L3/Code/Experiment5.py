import os
from tqdm import tqdm
from algorithms import JALGT
from solution_concepts import NashSolutionConcept
from game_model import GameModel
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from gymnasium import Wrapper
from pogema import pogema_v0, GridConfig
from pogema.animation import AnimationMonitor, AnimationConfig

def obs_to_state(obs):
    matrix_obstacles = obs[0]
    matrix_agents = obs[1]
    matrix_target = obs[2]
    target = np.max(matrix_target[2]) * 1 + matrix_target[1][0] * 2 + matrix_target[1][2] * 3
    obstacles = matrix_obstacles[0][1] * 2 ** 9 + matrix_obstacles[1][0] * 2 ** 8 + matrix_obstacles[1][2] * 2 ** 7 + matrix_obstacles[2][1] * 2 ** 6
    agents = matrix_agents[0][1] * 2 ** 5 + matrix_agents[1][0] * 2 ** 4 + matrix_agents[1][2] * 2 ** 3 + matrix_agents[2][1] * 2 ** 2
    return int(obstacles + agents + target)

class RewardWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
    def step(self, joint_action):
        observations, rewards, terminated, truncated, infos = self.env.step(joint_action)
        for i in range(len(joint_action)):
            if not terminated[i] and not truncated[i]:
                if rewards[i] == 0:
                    rewards[i] = rewards[i] - 0.01
        return observations, rewards, terminated, truncated, infos

def create_env(config, seed=42):
    grid_config = GridConfig(num_agents=config['num_agents'], size=config['size'], density=config['obstacle_density'], seed=seed, max_episode_steps=config['episode_length'], obs_radius=1, on_target='finish', render_mode=None)
    animation_config = AnimationConfig(directory=config['renders'], static=False, show_agents=True, egocentric_idx=None, save_every_idx_episode=None, show_border=True, show_lines=True)
    env = pogema_v0(grid_config)
    env = AnimationMonitor(env, animation_config=animation_config)
    return RewardWrapper(env)

def main():
    os.makedirs('Experimento5', exist_ok=True)
    SIZES = [4, 6, 8, 10]
    AGENTS = [2, 3, 4]
    OBSTACLE_DENSITY = 0.15
    EPOCHS = 200
    EPISODES_PER_EPOCH = 8
    results = []
    for size in SIZES:
        for n_agents in AGENTS:
            if n_agents > 2 and size != 4:
                continue
            if n_agents == 2 or (n_agents > 2 and size == 4):
                print(f'Tamaño: {size}x{size}, Agentes: {n_agents}')
                config = {
                    'size': size,
                    'num_agents': n_agents,
                    'obstacle_density': OBSTACLE_DENSITY,
                    'episode_length': size*2,
                    'renders': 'renders_exp5/',
                    'epochs': EPOCHS,
                    'episodes_per_epoch': EPISODES_PER_EPOCH,
                    'learning_rate': 0.01,
                    'epsilon_max': 1,
                    'epsilon_min': 0.1,
                    'num_states': 16 * 16 * 4,
                }
                game = GameModel(num_agents=n_agents, num_states=config['num_states'], num_actions=5)
                algorithms = [JALGT(i, game, NashSolutionConcept(), epsilon=config['epsilon_max'], alpha=config['learning_rate'], seed=i) for i in range(n_agents)]
                reward_per_epoch = []
                success_per_epoch = []
                import time
                t0 = time.time()
                for epoch in tqdm(range(EPOCHS)):
                    all_eval_rewards = []
                    all_success = []
                    for ep in range(EPISODES_PER_EPOCH):
                        env = create_env(config, seed=ep)
                        observations, infos = env.reset()
                        terminated = truncated = [False] * n_agents
                        total_rewards = [0] * n_agents
                        states = [obs_to_state(observations[i]) for i in range(n_agents)]
                        while not all(terminated) and not all(truncated):
                            actions = tuple([algorithms[i].select_action(states[i]) for i in range(n_agents)])
                            observations, rewards, terminated, truncated, infos = env.step(actions)
                            [algorithms[i].learn(actions, rewards, states[i], obs_to_state(observations[i])) for i in range(n_agents)]
                            total_rewards = [total_rewards[i] + rewards[i] for i in range(n_agents)]
                            states = [obs_to_state(observations[i]) for i in range(n_agents)]
                        all_eval_rewards.append(np.mean(total_rewards))
                        all_success.append(all(terminated))
                    reward_per_epoch.append(np.mean(all_eval_rewards))
                    success_per_epoch.append(np.mean(all_success))
                t1 = time.time()
                results.append({
                    'size': size,
                    'agents': n_agents,
                    'reward': reward_per_epoch,
                    'success': success_per_epoch,
                    'train_time': t1-t0
                })
    # Gráfica conjunta recompensa media por epoch (2 agentes)
    plt.figure(figsize=(12,5))
    for size in SIZES:
        for r in results:
            if r['size'] == size and r['agents'] == 2:
                plt.plot(r['reward'], label=f'{size}x{size}')
    plt.title('Recompensa media por epoch (2 agentes, todos los tamaños de mapa)')
    plt.xlabel('Epoch')
    plt.ylabel('Recompensa media')
    plt.legend()
    plt.savefig('Experimento5/recompensa_2agentes.png')
    plt.close()
    # Gráfica conjunta tasa de éxito por epoch (2 agentes)
    plt.figure(figsize=(12,5))
    for size in SIZES:
        for r in results:
            if r['size'] == size and r['agents'] == 2:
                plt.plot(r['success'], label=f'{size}x{size}')
    plt.title('Tasa de éxito por epoch (2 agentes, todos los tamaños de mapa)')
    plt.xlabel('Epoch')
    plt.ylabel('Tasa de éxito')
    plt.legend()
    plt.savefig('Experimento5/exito_2agentes.png')
    plt.close()
    # Gráficas individuales para 3 y 4 agentes (solo 4x4)
    for n_agents in [3, 4]:
        plt.figure(figsize=(12,5))
        for r in results:
            if r['size'] == 4 and r['agents'] == n_agents:
                plt.plot(r['reward'], label=f'{n_agents} agentes')
        plt.title(f'Recompensa media por epoch (4x4, {n_agents} agentes)')
        plt.xlabel('Epoch')
        plt.ylabel('Recompensa media')
        plt.legend()
        plt.savefig(f'Experimento5/recompensa_4x4_{n_agents}agentes.png')
        plt.close()
        plt.figure(figsize=(12,5))
        for r in results:
            if r['size'] == 4 and r['agents'] == n_agents:
                plt.plot(r['success'], label=f'{n_agents} agentes')
        plt.title(f'Tasa de éxito por epoch (4x4, {n_agents} agentes)')
        plt.xlabel('Epoch')
        plt.ylabel('Tasa de éxito')
        plt.legend()
        plt.savefig(f'Experimento5/exito_4x4_{n_agents}agentes.png')
        plt.close()
    # Tabla resumen
    summary = pd.DataFrame([{
        'Tamaño': r['size'],
        'Agentes': r['agents'],
        'Recompensa final': np.mean(r['reward'][-10:]),
        'Éxito final': np.mean(r['success'][-10:]),
        'Tiempo entrenamiento (s)': r['train_time']
    } for r in results])
    summary.to_csv('Experimento5/summary.csv', index=False)
    print('Gráficas y tabla resumen guardadas en Experimento5/')

if __name__ == "__main__":
    main()
