import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from algorithms import JALGT
from solution_concepts import ParetoSolutionConcept
from game_model import GameModel
import random
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
        previous_observations = self.env.unwrapped._obs()
        observations, rewards, terminated, truncated, infos = self.env.step(joint_action)
        for i in range(len(joint_action)):
            if not terminated[i] and not truncated[i]:
                if rewards[i] == 0:
                    rewards[i] = rewards[i] - 0.01
        return observations, rewards, terminated, truncated, infos

def create_env(config, seed=42):
    grid_config = GridConfig(num_agents=config['num_agents'], size=config['size'], density=config['obstacle_density'], seed=seed, max_episode_steps=config['episode_length'], obs_radius=1, on_target='finish', render_mode=None)
    animation_config = AnimationConfig(directory='renders/', static=False, show_agents=True, egocentric_idx=None, save_every_idx_episode=config['save_every'], show_border=True, show_lines=True)
    env = pogema_v0(grid_config)
    env = AnimationMonitor(env, animation_config=animation_config)
    return RewardWrapper(env)

def get_experiment_configs():
    exp_config = {
        'num_agents': 2,
        'size': 4,
        'maps': 10,
        'num_states': 16 * 16 * 4,
        'epochs': 200,
        'episodes_per_epoch': 10,
        'episode_length': 16,
        'obstacle_density': 0.1,
        'save_every': None,
        'learning_rate': 0.01,
        'epsilon_max': 1,
        'epsilon_min': 0.1,
        'renders': 'renders/',
        'solution_concept': ParetoSolutionConcept
    }
    return [
        dict(exp_config, num_agents=2, obstacle_density=0.1, learning_rate=0.01, solution_concept=ParetoSolutionConcept),
        dict(exp_config, num_agents=2, obstacle_density=0.3, learning_rate=0.01, solution_concept=ParetoSolutionConcept),
        dict(exp_config, num_agents=2, obstacle_density=0.5, learning_rate=0.01, solution_concept=ParetoSolutionConcept),
        dict(exp_config, num_agents=2, obstacle_density=0.1, learning_rate=0.05, solution_concept=ParetoSolutionConcept),        
        dict(exp_config, num_agents=2, obstacle_density=0.1, learning_rate=0.2, solution_concept=ParetoSolutionConcept),    
        dict(exp_config, num_agents=2, obstacle_density=0.1, learning_rate=0.5, solution_concept=ParetoSolutionConcept),    
        dict(exp_config, num_agents=2, obstacle_density=0.1, learning_rate=0.01, solution_concept=ParetoSolutionConcept),    
        dict(exp_config, num_agents=2, obstacle_density=0.1, learning_rate=0.01, epsilon_max=0.7 , solution_concept=ParetoSolutionConcept),    
        dict(exp_config, num_agents=2, obstacle_density=0.1, learning_rate=0.01, epsilon_max=0.5 , solution_concept=ParetoSolutionConcept),    
        dict(exp_config, num_agents=2, obstacle_density=0.1, learning_rate=0.01, epsilon_min=0.3 , solution_concept=ParetoSolutionConcept),
        dict(exp_config, num_agents=2, obstacle_density=0.1, learning_rate=0.01, epsilon_min=0.5 , solution_concept=ParetoSolutionConcept),
        dict(exp_config, num_agents=2, obstacle_density=0.1, learning_rate=0.01, epsilon_min=0.7 , solution_concept=ParetoSolutionConcept),
    ]

def main():
    os.makedirs('Experimento1', exist_ok=True)
    from collections import defaultdict
    import time
    results = defaultdict(lambda: {
        'reward_per_epoch': [],
        'td_error_per_epoch': [],
        'train_time_per_episode': [],
        'total_train_time': 0,
        'episodes': 0,
        'individual_reward': [],
        'collective_reward': [],
        'optimality': 0
    })
    for idx, config in enumerate(get_experiment_configs()):
        print(f"Ejecutando experimento {idx+1} con parámetros: { {k: v for k, v in config.items() if k not in ['solution_concept']} }")
        game = GameModel(num_agents=config['num_agents'], num_states=config['num_states'], num_actions=5)
        algorithms = [JALGT(i, game, config['solution_concept'](), epsilon=config['epsilon_max'], alpha=config['learning_rate'], seed=i) for i in range(game.num_agents)]
        epsilon_diff = (config['epsilon_max'] - config['epsilon_min']) / config['episodes_per_epoch']
        reward_per_epoch = []
        td_error_per_epoch = []
        train_time_per_episode = []
        individual_reward = []
        collective_reward = []
        start_total = time.time()
        for epoch in tqdm(range(config['epochs'])):
            all_eval_rewards = []
            all_td_errors = []
            for ep in range(config['episodes_per_epoch']):
                start_ep = time.time()
                env = create_env(config=config, seed=ep % config['maps'])
                observations, infos = env.reset()
                terminated = truncated = [False] * game.num_agents
                train_rewards = [0] * game.num_agents
                states = [obs_to_state(observations[i]) for i in range(game.num_agents)]
                while not all(terminated) and not all(truncated):
                    actions = tuple([algorithms[i].select_action(states[i]) for i in range(game.num_agents)])
                    observations, rewards, terminated, truncated, infos = env.step(actions)
                    [algorithms[i].learn(actions, rewards, states[i], obs_to_state(observations[i])) for i in range(game.num_agents)]
                    train_rewards = [train_rewards[i] + rewards[i] for i in range(game.num_agents)]
                    all_td_errors.append(algorithms[0].metrics['td_error'][-1])
                    states = [obs_to_state(observations[i]) for i in range(game.num_agents)]
                [algorithms[i].set_epsilon(config['epsilon_max'] - epsilon_diff * ep) for i in range(game.num_agents)]
                end_ep = time.time()
                train_time_per_episode.append(end_ep - start_ep)
                individual_reward.append(train_rewards[0])
                collective_reward.append(sum(train_rewards))
            td_error_per_epoch.append(sum(all_td_errors))
            evaluation_episodes = config['maps']
            all_eval_rewards = []
            for ep in range(evaluation_episodes):
                env = create_env(config=config, seed=ep)
                observations, infos = env.reset()
                terminated = truncated = [False] * game.num_agents
                total_rewards = [0] * config['num_agents']
                states = [obs_to_state(observations[i]) for i in range(game.num_agents)]
                while not all(terminated) and not all(truncated):
                    states = [obs_to_state(observations[i]) for i in range(game.num_agents)]
                    actions = tuple([algorithms[i].select_action(states[i], train=False) for i in range(game.num_agents)])
                    observations, rewards, terminated, truncated, infos = env.step(actions)
                    total_rewards = [total_rewards[i] + rewards[i] for i in range(config['num_agents'])]
                all_eval_rewards.append(sum(total_rewards))
            reward_per_epoch.append(sum(all_eval_rewards))
        end_total = time.time()
        optimality = np.mean([algorithms[0].value(0, s) for s in range(game.num_states)])
        label = f"agents={config['num_agents']}, obs_density={config['obstacle_density']}, lr={config['learning_rate']}, size={config.get('size', 4)}"
        results[label]['reward_per_epoch'] = reward_per_epoch
        results[label]['td_error_per_epoch'] = td_error_per_epoch
        results[label]['train_time_per_episode'] = train_time_per_episode
        results[label]['total_train_time'] = end_total - start_total
        results[label]['episodes'] = config['epochs'] * config['episodes_per_epoch']
        results[label]['individual_reward'] = individual_reward
        results[label]['collective_reward'] = collective_reward
        results[label]['optimality'] = optimality
    # Guardar gráficas
    plt.figure(figsize=(12, 6))
    for label, data in results.items():
        plt.plot(data['reward_per_epoch'], label=label)
    plt.title('Comparación de recompensa colectiva por experimento')
    plt.xlabel('Epoch')
    plt.ylabel('Recompensa colectiva')
    plt.legend()
    plt.savefig('Experimento1/recompensa_colectiva.png')
    plt.close()
    plt.figure(figsize=(12, 6))
    for label, data in results.items():
        plt.plot(data['td_error_per_epoch'], label=label)
    plt.title('Comparación de TD Error por experimento')
    plt.xlabel('Epoch')
    plt.ylabel('TD Error')
    plt.legend()
    plt.savefig('Experimento1/td_error.png')
    plt.close()
    plt.figure(figsize=(12, 6))
    for label, data in results.items():
        plt.plot(data['train_time_per_episode'], label=label)
    plt.title('Tiempo de entrenamiento por episodio')
    plt.xlabel('Episodio')
    plt.ylabel('Tiempo (s)')
    plt.legend()
    plt.savefig('Experimento1/tiempo_entrenamiento.png')
    plt.close()
    # Gráficas detalladas
    for idx, (label, data) in enumerate(results.items()):
        plt.figure(figsize=(12, 3))
        plt.plot(data['train_time_per_episode'])
        plt.title(f"Tiempo de entrenamiento por episodio\n{label}")
        plt.ylabel('Tiempo (s)')
        plt.xlabel('Episodio')
        plt.tight_layout()
        plt.savefig(f'Experimento1/tiempo_entrenamiento_{idx+1}.png')
        plt.close()
        plt.figure(figsize=(12, 3))
        plt.plot(data['individual_reward'])
        plt.title(f"Recompensa individual del agente 0\n{label}")
        plt.ylabel('Recompensa')
        plt.xlabel('Episodio')
        plt.tight_layout()
        plt.savefig(f'Experimento1/recompensa_individual_{idx+1}.png')
        plt.close()
        plt.figure(figsize=(12, 3))
        plt.plot(data['collective_reward'])
        plt.title(f"Recompensa colectiva por episodio\n{label}")
        plt.ylabel('Recompensa colectiva')
        plt.xlabel('Episodio')
        plt.tight_layout()
        plt.savefig(f'Experimento1/recompensa_colectiva_{idx+1}.png')
        plt.close()
    print('Gráficas guardadas en Experimento1/')

if __name__ == "__main__":
    main()
