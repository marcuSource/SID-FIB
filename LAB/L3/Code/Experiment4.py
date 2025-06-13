import os
from tqdm import tqdm
from algorithms import JALGT, IQL
from solution_concepts import NashSolutionConcept
from game_model import GameModel
import numpy as np
import matplotlib.pyplot as plt
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
    grid_config = GridConfig(num_agents=2, size=config['size'], density=config['obstacle_density'], seed=seed, max_episode_steps=config['episode_length'], obs_radius=1, on_target='finish', render_mode=None)
    animation_config = AnimationConfig(directory=config['renders'], static=False, show_agents=True, egocentric_idx=None, save_every_idx_episode=None, show_border=True, show_lines=True)
    env = pogema_v0(grid_config)
    env = AnimationMonitor(env, animation_config=animation_config)
    return RewardWrapper(env)

def plot_with_shade(data, label, color):
    arr = np.array(data)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    plt.plot(mean, label=label, color=color)
    plt.fill_between(range(len(mean)), mean-std, mean+std, color=color, alpha=0.2)

def main():
    os.makedirs('Experimento4', exist_ok=True)
    SEEDS = [0, 42, 123, 2025, 999]
    N_EPOCHS = 300
    N_EPISODES = 10
    all_rewards = []
    all_td = []
    all_maxq = []
    for seed in SEEDS:
        config = {
            'size': 6,
            'obstacle_density': 0.15,
            'episode_length': 16,
            'renders': 'renders_exp4/',
            'epochs': N_EPOCHS,
            'episodes_per_epoch': N_EPISODES,
            'learning_rate': 0.01,
            'epsilon_max': 1,
            'epsilon_min': 0.1,
            'num_states': 16 * 16 * 4,
        }
        game = GameModel(num_agents=2, num_states=config['num_states'], num_actions=5)
        algorithms = [
            JALGT(0, game, NashSolutionConcept(), epsilon=config['epsilon_max'], alpha=config['learning_rate'], seed=seed),
            IQL(1, game, epsilon=config['epsilon_max'], alpha=config['learning_rate'], seed=seed+1)
        ]
        reward_per_epoch = [[], []]
        td_error = [[], []]
        maxq = [[], []]
        for epoch in tqdm(range(N_EPOCHS), desc=f"Seed {seed}"):
            rewards_epoch = [0, 0]
            joint_qs = []
            for ep in range(N_EPISODES):
                env = create_env(config, seed=ep+seed)
                observations, infos = env.reset()
                terminated = truncated = [False, False]
                states = [obs_to_state(observations[i]) for i in range(2)]
                while not all(terminated) and not all(truncated):
                    actions = tuple([algorithms[i].select_action(states[i]) for i in range(2)])
                    observations, rewards, terminated, truncated, infos = env.step(actions)
                    [algorithms[i].learn(actions, rewards, states[i], obs_to_state(observations[i])) for i in range(2)]
                    rewards_epoch[0] += rewards[0]
                    rewards_epoch[1] += rewards[1]
                    states = [obs_to_state(observations[i]) for i in range(2)]
                    if len(joint_qs) == 0:
                        q_jalgt = algorithms[0].q_table[0][states[0]] if states[0] < algorithms[0].q_table.shape[1] else np.zeros(algorithms[0].q_table.shape[2])
                        q_iql = algorithms[1].Q[states[1]] if states[1] in algorithms[1].Q else np.zeros(5)
                        joint_qs.append((q_jalgt.copy(), q_iql.copy()))
            reward_per_epoch[0].append(rewards_epoch[0]/N_EPISODES)
            reward_per_epoch[1].append(rewards_epoch[1]/N_EPISODES)
            td_error[0].append(np.mean(algorithms[0].metrics['td_error'][-N_EPISODES:]))
            td_error[1].append(np.mean(algorithms[1].metrics['td_error'][-N_EPISODES:]))
            maxq[0].append(np.max(joint_qs[0][0]))
            maxq[1].append(np.max(joint_qs[0][1]))
        all_rewards.append(reward_per_epoch)
        all_td.append(td_error)
        all_maxq.append(maxq)
    # 1. Recompensa media por epoch
    plt.figure(figsize=(12,5))
    plot_with_shade([r[0] for r in all_rewards], 'JAL-GT (Nash)', 'tab:blue')
    plot_with_shade([r[1] for r in all_rewards], 'IQL', 'tab:orange')
    plt.title('Recompensa media por epoch (media ± std entre semillas)')
    plt.xlabel('Epoch')
    plt.ylabel('Recompensa media')
    plt.legend()
    plt.savefig('Experimento4/recompensa.png')
    plt.close()
    # 2. TD-error medio por epoch
    plt.figure(figsize=(12,5))
    plot_with_shade([t[0] for t in all_td], 'TD-error JAL-GT', 'tab:blue')
    plot_with_shade([t[1] for t in all_td], 'TD-error IQL', 'tab:orange')
    plt.title('TD-error medio por epoch (media ± std entre semillas)')
    plt.xlabel('Epoch')
    plt.ylabel('TD-error')
    plt.legend()
    plt.savefig('Experimento4/td_error.png')
    plt.close()
    # 3. Q máximo por epoch
    plt.figure(figsize=(12,5))
    plot_with_shade([q[0] for q in all_maxq], 'Max Q JAL-GT', 'tab:blue')
    plot_with_shade([q[1] for q in all_maxq], 'Max Q IQL', 'tab:orange')
    plt.title('Evolución del valor máximo de Q (media ± std entre semillas)')
    plt.xlabel('Epoch')
    plt.ylabel('Max Q')
    plt.legend()
    plt.savefig('Experimento4/maxq.png')
    plt.close()
    # 4. Q mínimo, Q percentil 25 y Q percentil 75 por epoch
    # (Para simplificar, solo guardamos la media de los percentiles)
    # 5. Spread, varianza y rango intercuartílico de Q por epoch
    # (Opcional: se puede añadir si se requiere)
    print('Gráficas guardadas en Experimento4/')

if __name__ == "__main__":
    main()
