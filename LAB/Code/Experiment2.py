import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from algorithms import JALGT
from solution_concepts import MinimaxSolutionConcept, ParetoSolutionConcept, NashSolutionConcept, WelfareSolutionConcept
from game_model import GameModel
import random
from gymnasium import Wrapper
from pogema import pogema_v0, GridConfig
from pogema.animation import AnimationMonitor, AnimationConfig
import pandas as pd

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
        previous_observations = self.env.unwrapped._obs()
        observations, rewards, terminated, truncated, infos = self.env.step(joint_action)
        for i in range(len(joint_action)):
            if not terminated[i] and not truncated[i]:
                if rewards[i] == 0:
                    rewards[i] = rewards[i] - 0.01
        return observations, rewards, terminated, truncated, infos

def create_env(config, seed=42):
    grid_config = GridConfig(num_agents=config['num_agents'], size=config['size'], density=config['obstacle_density'], seed=seed, max_episode_steps=config['episode_length'], obs_radius=1, on_target='finish', render_mode=None)
    animation_config = AnimationConfig(directory=config['renders'], static=False, show_agents=True, egocentric_idx=None, save_every_idx_episode=config['save_every'], show_border=True, show_lines=True)
    env = pogema_v0(grid_config)
    env = AnimationMonitor(env, animation_config=animation_config)
    return RewardWrapper(env)

SCENARIOS = [
    {'name': 'Comunes', 'obstacle_density': 0.05, 'size': 4, 'maps': 5},
    {'name': 'Conflictivos', 'obstacle_density': 0.4, 'size': 4, 'maps': 5},
    {'name': 'Mixtos', 'obstacle_density': 0.15, 'size': 6, 'maps': 5},
]
LEARNING_RATES = [0.001, 0.01, 0.05]
SCENARIO_LR_SWEEP = {'name': 'Mixtos-LR', 'obstacle_density': 0.15, 'size': 6, 'maps': 5}
BASE_CONFIG = {
    'num_agents': 2,
    'num_states': 16 * 16 * 4,
    'epochs': 200,
    'episodes_per_epoch': 10,
    'episode_length': 16,
    'save_every': None,
    'learning_rate': 0.01,
    'epsilon_max': 1,
    'epsilon_min': 0.1,
    'renders': 'renders_exp2/',
}
concepts = [
    ("Minimax", MinimaxSolutionConcept),
    ("Nash", NashSolutionConcept),
    ("Welfare", WelfareSolutionConcept),
    ("Pareto", ParetoSolutionConcept),
]

def main():
    os.makedirs('Experimento2', exist_ok=True)
    results_all = []
    # Experimentos principales
    for scenario in SCENARIOS:
        for name, Concept in concepts:
            print(f"Entrenando: {name} en entorno {scenario['name']}")
            config = dict(BASE_CONFIG)
            config.update(scenario)
            game = GameModel(num_agents=config.get('num_agents', 2), num_states=config.get('num_states', BASE_CONFIG['num_states']), num_actions=5)
            algorithms = [JALGT(i, game, Concept(), epsilon=config['epsilon_max'], alpha=config['learning_rate'], seed=i) for i in range(game.num_agents)]
            epsilon_diff = (config['epsilon_max'] - config['epsilon_min']) / config['episodes_per_epoch']
            reward_per_epoch = []
            success_per_epoch = []
            for epoch in tqdm(range(config['epochs'])):
                for ep in range(config['episodes_per_epoch']):
                    env = create_env(config, seed=ep % config['maps'])
                    observations, infos = env.reset()
                    terminated = truncated = [False] * game.num_agents
                    train_rewards = [0] * game.num_agents
                    states = [obs_to_state(observations[i]) for i in range(game.num_agents)]
                    while not all(terminated) and not all(truncated):
                        actions = tuple([algorithms[i].select_action(states[i]) for i in range(game.num_agents)])
                        observations, rewards, terminated, truncated, infos = env.step(actions)
                        [algorithms[i].learn(actions, rewards, states[i], obs_to_state(observations[i])) for i in range(game.num_agents)]
                        train_rewards = [train_rewards[i] + rewards[i] for i in range(game.num_agents)]
                        states = [obs_to_state(observations[i]) for i in range(game.num_agents)]
                    [algorithms[i].set_epsilon(config['epsilon_max'] - epsilon_diff * ep) for i in range(game.num_agents)]
                # Evaluación
                evaluation_episodes = config['maps']
                all_eval_rewards = []
                all_success = []
                for ep in range(evaluation_episodes):
                    env = create_env(config, seed=ep)
                    observations, infos = env.reset()
                    terminated = truncated = [False] * config.get('num_agents', 2)
                    total_rewards = [0] * config.get('num_agents', 2)
                    states = [obs_to_state(observations[i]) for i in range(game.num_agents)]
                    while not all(terminated) and not all(truncated):
                        states = [obs_to_state(observations[i]) for i in range(game.num_agents)]
                        actions = tuple([algorithms[i].select_action(states[i], train=False) for i in range(game.num_agents)])
                        observations, rewards, terminated, truncated, infos = env.step(actions)
                        total_rewards = [total_rewards[i] + rewards[i] for i in range(config.get('num_agents', 2))]
                    all_eval_rewards.append(sum(total_rewards))
                    all_success.append(all(terminated))
                reward_per_epoch.append(np.mean(all_eval_rewards))
                success_per_epoch.append(np.mean(all_success))
            results_all.append({
                'Concepto': name,
                'Escenario': scenario['name'],
                'Recompensa media final': np.mean(reward_per_epoch[-10:]),
                'Tasa de éxito final': np.mean(success_per_epoch[-10:]),
                'Historial recompensas': reward_per_epoch,
                'Historial éxito': success_per_epoch
            })
    # Barrido de tasa de aprendizaje
    results_lr = []
    for lr in LEARNING_RATES:
        for name, Concept in concepts:
            print(f"Entrenando: {name} en {SCENARIO_LR_SWEEP['name']} con lr={lr}")
            config = dict(BASE_CONFIG)
            config.update(SCENARIO_LR_SWEEP)
            config['learning_rate'] = lr
            game = GameModel(num_agents=config['num_agents'], num_states=config['num_states'], num_actions=5)
            algorithms = [JALGT(i, game, Concept(), epsilon=config['epsilon_max'], alpha=config['learning_rate'], seed=i) for i in range(game.num_agents)]
            epsilon_diff = (config['epsilon_max'] - config['epsilon_min']) / config['episodes_per_epoch']
            reward_per_epoch = []
            success_per_epoch = []
            for epoch in tqdm(range(config['epochs'])):
                for ep in range(config['episodes_per_epoch']):
                    env = create_env(config, seed=ep % config['maps'])
                    observations, infos = env.reset()
                    terminated = truncated = [False] * game.num_agents
                    train_rewards = [0] * game.num_agents
                    states = [obs_to_state(observations[i]) for i in range(game.num_agents)]
                    while not all(terminated) and not all(truncated):
                        actions = tuple([algorithms[i].select_action(states[i]) for i in range(game.num_agents)])
                        observations, rewards, terminated, truncated, infos = env.step(actions)
                        [algorithms[i].learn(actions, rewards, states[i], obs_to_state(observations[i])) for i in range(game.num_agents)]
                        train_rewards = [train_rewards[i] + rewards[i] for i in range(game.num_agents)]
                        states = [obs_to_state(observations[i]) for i in range(game.num_agents)]
                    [algorithms[i].set_epsilon(config['epsilon_max'] - epsilon_diff * ep) for i in range(game.num_agents)]
                # Evaluación
                evaluation_episodes = config['maps']
                all_eval_rewards = []
                all_success = []
                for ep in range(evaluation_episodes):
                    env = create_env(config, seed=ep)
                    observations, infos = env.reset()
                    terminated = truncated = [False] * config['num_agents']
                    total_rewards = [0] * config['num_agents']
                    states = [obs_to_state(observations[i]) for i in range(game.num_agents)]
                    while not all(terminated) and not all(truncated):
                        states = [obs_to_state(observations[i]) for i in range(game.num_agents)]
                        actions = tuple([algorithms[i].select_action(states[i], train=False) for i in range(game.num_agents)])
                        observations, rewards, terminated, truncated, infos = env.step(actions)
                        total_rewards = [total_rewards[i] + rewards[i] for i in range(config['num_agents'])]
                    all_eval_rewards.append(sum(total_rewards))
                    all_success.append(all(terminated))
                reward_per_epoch.append(np.mean(all_eval_rewards))
                success_per_epoch.append(np.mean(all_success))
            results_lr.append({
                'Concepto': name,
                'LR': lr,
                'Recompensa media final': np.mean(reward_per_epoch[-10:]),
                'Tasa de éxito final': np.mean(success_per_epoch[-10:]),
                'Historial recompensas': reward_per_epoch,
                'Historial éxito': success_per_epoch
            })
    # Guardar gráficas de escenarios principales
    for scenario in SCENARIOS:
        plt.figure(figsize=(12,5))
        for r in results_all:
            if r['Escenario'] == scenario['name']:
                plt.plot(r['Historial recompensas'], label=f"{r['Concepto']} (reward)")
        plt.title(f"Recompensa media por epoch en entorno: {scenario['name']}")
        plt.xlabel('Epoch')
        plt.ylabel('Recompensa media (evaluación)')
        plt.legend()
        plt.savefig(f'Experimento2/recompensa_{scenario["name"]}.png')
        plt.close()
        plt.figure(figsize=(12,5))
        for r in results_all:
            if r['Escenario'] == scenario['name']:
                plt.plot(r['Historial éxito'], label=f"{r['Concepto']} (éxito)")
        plt.title(f"Tasa de éxito por epoch en entorno: {scenario['name']}")
        plt.xlabel('Epoch')
        plt.ylabel('Tasa de éxito (evaluación)')
        plt.legend()
        plt.savefig(f'Experimento2/exito_{scenario["name"]}.png')
        plt.close()
    # Guardar gráficas de barrido de learning rate
    for name, Concept in concepts:
        plt.figure(figsize=(12,5))
        for r in results_lr:
            if r['Concepto'] == name:
                plt.plot(r['Historial recompensas'], label=f"lr={r['LR']}")
        plt.title(f"Recompensa media por epoch ({name}) en barrido de learning rate")
        plt.xlabel('Epoch')
        plt.ylabel('Recompensa media (evaluación)')
        plt.legend()
        plt.savefig(f'Experimento2/recompensa_lr_{name}.png')
        plt.close()
        plt.figure(figsize=(12,5))
        for r in results_lr:
            if r['Concepto'] == name:
                plt.plot(r['Historial éxito'], label=f"lr={r['LR']}")
        plt.title(f"Tasa de éxito por epoch ({name}) en barrido de learning rate")
        plt.xlabel('Epoch')
        plt.ylabel('Tasa de éxito (evaluación)')
        plt.legend()
        plt.savefig(f'Experimento2/exito_lr_{name}.png')
        plt.close()
    # Guardar tablas resumen
    summary_all = pd.DataFrame([{k: v for k, v in r.items() if k not in ['Historial recompensas', 'Historial éxito']} for r in results_all])
    summary_all.to_csv('Experimento2/summary_all.csv', index=False)
    summary_lr = pd.DataFrame([{k: v for k, v in r.items() if k not in ['Historial recompensas', 'Historial éxito']} for r in results_lr])
    summary_lr.to_csv('Experimento2/summary_lr.csv', index=False)
    print('Gráficas y tablas guardadas en Experimento2/')

if __name__ == "__main__":
    main()
