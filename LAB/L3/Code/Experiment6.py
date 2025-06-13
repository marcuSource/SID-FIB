import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import time
from algorithms import JALGT
from solution_concepts import NashSolutionConcept
from game_model import GameModel
from pogema import pogema_v0, GridConfig
from pogema.animation import AnimationMonitor, AnimationConfig
from gymnasium import Wrapper

# Output directory
OUTDIR = 'Experimento6'
os.makedirs(OUTDIR, exist_ok=True)

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

def obs_to_state(obs):
    matrix_obstacles = obs[0]
    matrix_agents = obs[1]
    matrix_target = obs[2]
    target = np.max(matrix_target[2]) * 1 + matrix_target[1][0] * 2 + matrix_target[1][2] * 3
    obstacles = matrix_obstacles[0][1] * 2 ** 9 + matrix_obstacles[1][0] * 2 ** 8 + matrix_obstacles[1][2] * 2 ** 7 + matrix_obstacles[2][1] * 2 ** 6
    agents = matrix_agents[0][1] * 2 ** 5 + matrix_agents[1][0] * 2 ** 4 + matrix_agents[1][2] * 2 ** 3 + matrix_agents[2][1] * 2 ** 2
    return int(obstacles + agents + target)

# Configuración
def main():
    SIZES = [6]
    AGENTS = [2]
    OBSTACLE_DENSITY = 0.1
    EPOCHS = 500
    EPISODES_PER_EPOCH = 10
    N_TEST_MAPS = 10
    TRAIN_SEEDS = list(range(100))
    TEST_SEEDS = random.sample(TRAIN_SEEDS, N_TEST_MAPS)
    TRAIN_SEEDS = [s for s in TRAIN_SEEDS if s not in TEST_SEEDS]

    config = {
        'size': SIZES[0],
        'num_agents': AGENTS[0],
        'obstacle_density': OBSTACLE_DENSITY,
        'episode_length': SIZES[0]*2,
        'renders': os.path.join(OUTDIR, 'renders_exp6/'),
        'epochs': EPOCHS,
        'episodes_per_epoch': EPISODES_PER_EPOCH,
        'learning_rate': 0.01,
        'epsilon_max': 1,
        'epsilon_min': 0.1,
        'num_states': 16 * 16 * 4,
    }
    os.makedirs(config['renders'], exist_ok=True)
    game = GameModel(num_agents=AGENTS[0], num_states=config['num_states'], num_actions=5)
    algorithms = [JALGT(i, game, NashSolutionConcept(), epsilon=config['epsilon_max'], alpha=config['learning_rate'], seed=i) for i in range(AGENTS[0])]
    rewards_train = []
    success_train = []
    t0 = time.time()
    for epoch in range(EPOCHS):
        epoch_rewards = []
        epoch_success = []
        for ep in range(EPISODES_PER_EPOCH):
            seed = random.choice(TRAIN_SEEDS)
            env = create_env(config, seed=seed)
            observations, infos = env.reset()
            terminated = truncated = [False] * AGENTS[0]
            total_rewards = [0] * AGENTS[0]
            states = [obs_to_state(observations[i]) for i in range(AGENTS[0])]
            while not all(terminated) and not all(truncated):
                actions = tuple([algorithms[i].select_action(states[i]) for i in range(AGENTS[0])])
                observations, rewards, terminated, truncated, infos = env.step(actions)
                [algorithms[i].learn(actions, rewards, states[i], obs_to_state(observations[i])) for i in range(AGENTS[0])]
                total_rewards = [total_rewards[i] + rewards[i] for i in range(AGENTS[0])]
                states = [obs_to_state(observations[i]) for i in range(AGENTS[0])]
            epoch_rewards.append(np.sum(total_rewards))
            epoch_success.append(all(terminated))
        rewards_train.append(np.mean(epoch_rewards))
        success_train.append(np.mean(epoch_success))
    t1 = time.time()
    train_time = t1-t0

    # Evaluación en mapas no vistos y vistos
    rewards_test = []
    success_test = []
    success_test_individual = []
    for seed in TEST_SEEDS:
        env = create_env(config, seed=seed)
        observations, infos = env.reset()
        terminated = truncated = [False] * AGENTS[0]
        total_rewards = [0] * AGENTS[0]
        states = [obs_to_state(observations[i]) for i in range(AGENTS[0])]
        while not all(terminated) and not all(truncated):
            actions = tuple([algorithms[i].select_action(states[i]) for i in range(AGENTS[0])])
            observations, rewards, terminated, truncated, infos = env.step(actions)
            total_rewards = [total_rewards[i] + rewards[i] for i in range(AGENTS[0])]
            states = [obs_to_state(observations[i]) for i in range(AGENTS[0])]
        rewards_test.append(np.sum(total_rewards))
        success_test.append(all(terminated))
        success_test_individual.append(np.mean(terminated))
    N_EVAL = len(TEST_SEEDS)
    rewards_seen = []
    success_seen = []
    success_seen_individual = []
    for seed in TRAIN_SEEDS[:N_EVAL]:
        env = create_env(config, seed=seed)
        observations, infos = env.reset()
        terminated = truncated = [False] * AGENTS[0]
        total_rewards = [0] * AGENTS[0]
        states = [obs_to_state(observations[i]) for i in range(AGENTS[0])]
        while not all(terminated) and not all(truncated):
            actions = tuple([algorithms[i].select_action(states[i]) for i in range(AGENTS[0])])
            observations, rewards, terminated, truncated, infos = env.step(actions)
            total_rewards = [total_rewards[i] + rewards[i] for i in range(AGENTS[0])]
            states = [obs_to_state(observations[i]) for i in range(AGENTS[0])]
        rewards_seen.append(np.sum(total_rewards))
        success_seen.append(all(terminated))
        success_seen_individual.append(np.mean(terminated))

    # Gráfica de recompensa colectiva final
    plt.figure(figsize=(10,5))
    plt.bar(['Entrenamiento (últimos 10)', 'Test no vistos', 'Test vistos'],
            [np.mean(rewards_train[-10:]), np.mean(rewards_test), np.mean(rewards_seen)],
            color=['blue', 'red', 'green'])
    plt.ylabel('Recompensa colectiva final')
    plt.title('Comparación de recompensa colectiva final')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'recompensa_colectiva_final.png'))
    plt.close()

    # Gráfica de tasa de éxito final
    plt.figure(figsize=(10,5))
    plt.bar(['Entrenamiento (últimos 10)', 'Test no vistos', 'Test vistos'],
            [np.mean(success_train[-10:]), np.mean(success_test), np.mean(success_seen)],
            color=['blue', 'red', 'green'])
    plt.ylabel('Tasa de éxito')
    plt.title('Comparación de tasa de éxito final')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'tasa_exito_final.png'))
    plt.close()

    # Gráfica de tasa de éxito individual final
    plt.figure(figsize=(10,5))
    plt.bar(['Test no vistos', 'Test vistos'],
            [np.mean(success_test_individual), np.mean(success_seen_individual)],
            color=['red', 'green'])
    plt.ylabel('Tasa de éxito individual')
    plt.title('Comparación de tasa de éxito individual (por agente)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'tasa_exito_individual_final.png'))
    plt.close()

    # Gráfica de tasa de éxito individual durante el entrenamiento
    success_train_individual = []
    for epoch in range(EPOCHS):
        epoch_success_ind = []
        for ep in range(EPISODES_PER_EPOCH):
            seed = random.choice(TRAIN_SEEDS)
            env = create_env(config, seed=seed)
            observations, infos = env.reset()
            terminated = truncated = [False] * AGENTS[0]
            states = [obs_to_state(observations[i]) for i in range(AGENTS[0])]
            while not all(terminated) and not all(truncated):
                actions = tuple([algorithms[i].select_action(states[i]) for i in range(AGENTS[0])])
                observations, rewards, terminated, truncated, infos = env.step(actions)
                states = [obs_to_state(observations[i]) for i in range(AGENTS[0])]
            epoch_success_ind.append(np.mean(terminated))
        success_train_individual.append(np.mean(epoch_success_ind))

    plt.figure(figsize=(10,5))
    plt.plot(success_train_individual, label='Entrenamiento (por agente)')
    plt.axhline(np.mean(success_test_individual), color='red', linestyle='--', label='Test no vistos (media)')
    plt.axhline(np.mean(success_seen_individual), color='green', linestyle='--', label='Test vistos (media)')
    plt.ylabel('Tasa de éxito individual')
    plt.xlabel('Epoch')
    plt.title('Tasa de éxito individual durante entrenamiento y test')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'tasa_exito_individual_entrenamiento.png'))
    plt.close()

    # Tabla resumen
    summary = pd.DataFrame({
        'Fase': ['Entrenamiento (últimos 10)', 'Test no vistos', 'Test vistos'],
        'Recompensa colectiva': [np.mean(rewards_train[-10:]), np.mean(rewards_test), np.mean(rewards_seen)],
        'Tasa de éxito': [np.mean(success_train[-10:]), np.mean(success_test), np.mean(success_seen)],
        'Tasa de éxito individual': [None, np.mean(success_test_individual), np.mean(success_seen_individual)],
        'Tiempo entrenamiento (s)': [train_time, None, None]
    })
    summary.to_csv(os.path.join(OUTDIR, 'resumen.csv'), index=False)

if __name__ == '__main__':
    main()
