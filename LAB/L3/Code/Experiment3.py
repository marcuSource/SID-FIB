import os
from tqdm import tqdm
from algorithms import JALGT
from solution_concepts import MinimaxSolutionConcept, ParetoSolutionConcept, NashSolutionConcept, WelfareSolutionConcept
from game_model import GameModel
import numpy as np
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
    animation_config = AnimationConfig(directory=config['renders'], static=False, show_agents=True, egocentric_idx=None, save_every_idx_episode=config['save_every'], show_border=True, show_lines=True)
    env = pogema_v0(grid_config)
    env = AnimationMonitor(env, animation_config=animation_config)
    return RewardWrapper(env)

def main():
    exp_config = {
        'num_agents': 2,
        'size': 4,
        'maps': 4,
        'num_states': 16 * 16 * 4,
        'epochs': 200,
        'episodes_per_epoch': 10,
        'episode_length': 16,
        'obstacle_density': 0.1,
        'save_every': 1,
        'learning_rate': 0.01,
        'epsilon_max': 1,
        'epsilon_min': 0.1,
        'renders': 'Experimento3/',
    }
    os.makedirs(exp_config['renders'], exist_ok=True)
    solution_concepts = [
        ("Minimax", MinimaxSolutionConcept),
        ("Nash", NashSolutionConcept),
        ("Welfare", WelfareSolutionConcept),
        ("Pareto", ParetoSolutionConcept),
    ]
    concept_pairs = [
        (("Minimax", MinimaxSolutionConcept), ("Nash", NashSolutionConcept)),
        (("Minimax", MinimaxSolutionConcept), ("Welfare", WelfareSolutionConcept)),
        (("Minimax", MinimaxSolutionConcept), ("Pareto", ParetoSolutionConcept)),
        (("Nash", NashSolutionConcept), ("Welfare", WelfareSolutionConcept)),
        (("Nash", NashSolutionConcept), ("Pareto", ParetoSolutionConcept)),
        (("Welfare", WelfareSolutionConcept), ("Pareto", ParetoSolutionConcept)),
    ]
    for (name1, concept1), (name2, concept2) in concept_pairs:
        print(f"Entrenando: Agente 0 = {name1}, Agente 1 = {name2}")
        game = GameModel(num_agents=2, num_states=exp_config['num_states'], num_actions=5)
        algorithms = [
            JALGT(0, game, concept1(), epsilon=exp_config['epsilon_max'], alpha=exp_config['learning_rate'], seed=0),
            JALGT(1, game, concept2(), epsilon=exp_config['epsilon_max'], alpha=exp_config['learning_rate'], seed=1)
        ]
        epsilon_diff = (exp_config['epsilon_max'] - exp_config['epsilon_min']) / exp_config['episodes_per_epoch']
        for epoch in tqdm(range(exp_config['epochs'])):
            for ep in range(exp_config['episodes_per_epoch']):
                env = create_env(exp_config, seed=ep % exp_config['maps'])
                observations, infos = env.reset()
                terminated = truncated = [False] * 2
                states = [obs_to_state(observations[i]) for i in range(2)]
                while not all(terminated) and not all(truncated):
                    actions = tuple([algorithms[i].select_action(states[i]) for i in range(2)])
                    observations, rewards, terminated, truncated, infos = env.step(actions)
                    [algorithms[i].learn(actions, rewards, states[i], obs_to_state(observations[i])) for i in range(2)]
                    states = [obs_to_state(observations[i]) for i in range(2)]
                [algorithms[i].set_epsilon(exp_config['epsilon_max'] - epsilon_diff * ep) for i in range(2)]
        # Animar una partida de evaluación para cada mapa
        for eval_map in range(exp_config['maps']):
            env = create_env(exp_config, seed=eval_map)
            observations, infos = env.reset()
            terminated = truncated = [False] * 2
            states = [obs_to_state(observations[i]) for i in range(2)]
            while not all(terminated) and not all(truncated):
                actions = tuple([algorithms[i].select_action(states[i], train=False) for i in range(2)])
                observations, rewards, terminated, truncated, infos = env.step(actions)
                states = [obs_to_state(observations[i]) for i in range(2)]
            anim_file = f"Experimento3/Anim_{name1}_vs_{name2}_map{eval_map}_global.svg"
            env.save_animation(anim_file, AnimationConfig(egocentric_idx=None, show_border=True, show_lines=True))
            print(f"Animación guardada: {anim_file}")
    print('Animaciones guardadas en Experimento3/')

if __name__ == "__main__":
    main()
