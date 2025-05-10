SLIPPERY = True

##EXPERIMENTACIÓ AMB
#NUMERO D'EPISODIS A ENTRENAR
#FACTOR DE DESCOMPTE
#SENYAL DE RECOMPENSA
#EPSILON I EPSILON DECAY
#LEARNING RATE I LEARNING DECAY

##EVALUACIÓ D'ENTRENAMENTS
#TEMPS D'ENTRENAMENT PER EPISODI
#NUMERO EPISODIS
#TEMPS TOTAL
#RECOMPENSA OBTINGUDA
#OPTIMALITAT DE POLITICA RESULTANT

#VALORS ESTANDARD
T_MAX = 100               # Aumentar para permitir más pasos en episodios largos
NUM_EPISODES = 2000       # Necesario para que aprenda bien
GAMMA = 0.99              # Alta importancia a las recompensas futuras
REWARD_THRESHOLD = 0.78   # FrozenLake es muy difícil de resolver al 100%
LEARNING_RATE = 0.8       # Aprendizaje agresivo al principio
EPSILON = 1.0             # Alta exploración al inicio

EPSILON_DECAY = 0.99
EPSILON_MIN = 0.01
LEARNING_RATE_DECAY = 0.999


import gymnasium as gym
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from gymnasium import Wrapper
from itertools import product
import time

## DEFINICIONS PRINCIPALS d'entorn:

env = gym.make("CliffWalking-v0", is_slippery=SLIPPERY)

def test_episode(agent, env):
    env.reset()
    is_done = False
    t = 0

    while not is_done:
        action = agent.select_action()
        state, reward, is_done, truncated, info = env.step(action)
        t += 1
    return state, reward, is_done, truncated, info

def draw_rewards(rewards):
    data = pd.DataFrame({'Episode': range(1, len(rewards) + 1), 'Reward': rewards})
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Episode', y='Reward', data=data)

    plt.title('Rewards Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.tight_layout()

    plt.show()
    
def print_policy(policy):
    visual_help = {0:'<', 1:'v', 2:'>', 3:'^'}
    policy_arrows = [visual_help[x] for x in policy]
    print(np.array(policy_arrows).reshape([-1, 4]))

##AGENT
class QLearningAgent:
    def __init__(self, env, gamma, learning_rate, epsilon, t_max,
                 epsilon_decay, learning_rate_decay, epsilon_min):
        self.env = env
        self.Q = np.zeros((env.observation_space.n, env.action_space.n))
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.t_max = t_max
        self.epsilon_decay = epsilon_decay
        self.learning_rate_decay = learning_rate_decay
        self.epsilon_min = epsilon_min

    def select_action(self, state, training=True):
        if training and random.random() <= self.epsilon:
            return np.random.choice(self.env.action_space.n)
        else:
            return np.argmax(self.Q[state])

    def update_Q(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.Q[next_state])
        td_target = reward + self.gamma * self.Q[next_state, best_next_action]
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.learning_rate * td_error

    def learn_from_episode(self):
        state, _ = self.env.reset()
        total_reward = 0
        for i in range(self.t_max):
            action = self.select_action(state)
            new_state, new_reward, is_done, truncated, _ = self.env.step(action)
            total_reward += new_reward
            self.update_Q(state, action, new_reward, new_state)
            if is_done:
                break
            state = new_state

        # Decay epsilon and learning rate
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.learning_rate *= self.learning_rate_decay

        return total_reward

    def policy(self):
        policy = np.zeros(self.env.observation_space.n, dtype=int)
        for s in range(self.env.observation_space.n):
            policy[s] = np.argmax(self.Q[s])
        return policy
    
class CustomCliffWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        state, reward, is_done, truncated, info = self.env.step(action)
         # Penalización por cada paso para promover caminos más cortos
        reward -= 0.01

        # Bonificación si llega al objetivo (el reward original es 1.0)
        if reward == 1.0:
            reward += 0.1  # Bonus por alcanzar el objetivo

        # Penalización extra si termina el episodio sin llegar al objetivo
        if is_done and reward < 1.0:
            reward -= 1  # Castigo por caer en un agujero o fallar
        return state, reward, is_done, truncated, info

def visualitza():

    # Carrega dels resultats
    df = pd.read_csv("experiments_Qlearning_results.csv")

        
    # Expandeix la columna 'config' que conté un diccionari en columnes separades
    config_df = df['config'].apply(eval).apply(pd.Series)
    df = pd.concat([config_df, df.drop(columns=['config'])], axis=1)

    # Configuració gràfica
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    fig.suptitle("Resultats dels Experiments Q-Learning", fontsize=16)

    # ─── Gràfic 1: Reward segons GAMMA i LEARNING_RATE ─────────────
    sns.lineplot(
        data=df,
        x="GAMMA",
        y="mean_last100_reward",
        hue="LEARNING_RATE",
        marker="o",
        ax=axes[0, 0]
    )
    axes[0, 0].set_title("Reward mitjana (últims 100 episodis)")
    axes[0, 0].set_xlabel("Gamma")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].legend(title="Learning Rate")

    # ─── Gràfic 2: Temps d'entrenament segons número d'episodis ─────
    sns.boxplot(
        data=df,
        x="NUM_EPISODES",
        y="training_time",
        ax=axes[0, 1]
    )
    axes[0, 1].set_title("Temps total d'entrenament")
    axes[0, 1].set_xlabel("Número d'episodis")
    axes[0, 1].set_ylabel("Temps (s)")

    # ─── Gràfic 3: Òptimes accions segons EPSILON_DECAY ─────────────
    sns.barplot(
        data=df,
        x="EPSILON_DECAY",
        y="optimality_ratio",
        ci="sd",
        ax=axes[1, 0]
    )
    axes[1, 0].set_title("Proporció d'accions òptimes")
    axes[1, 0].set_xlabel("Epsilon Decay")
    axes[1, 0].set_ylabel("Òptimes / Total")

    # ─── Gràfic 4: Heatmap combinació Gamma - LR vs Reward ─────────
    pivot_table = df.pivot_table(
        index="GAMMA",
        columns="LEARNING_RATE",
        values="mean_last100_reward",
        aggfunc="mean"
    )

    sns.heatmap(
        pivot_table,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        ax=axes[1, 1]
    )
    axes[1, 1].set_title("Reward segons Gamma i Learning Rate")
    axes[1, 1].set_xlabel("Learning Rate")
    axes[1, 1].set_ylabel("Gamma")

    # ─── Ajustaments finals ─────────────────────────────────────────
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # espai per al títol principal
    plt.show()


execucio = int(input("Prem 0 si només vols veure els resultats dels entrenaments, Prem 1 per executar un entrenament estandard, Prem qualsevol altre valor per executar els experiments (despres s'en visualitzaran els resultats)"))
if(execucio == 0):
    visualitza()

elif(execucio == 1):

    #EXECUCIÓ DE PROVA AMB VALORS ESTANDARDS I WRAPPER (MODIFICACIÓ DE REWARDS I PENALITZACIONS)

    fixed_env = CustomCliffWrapper(env)
    agent = QLearningAgent(fixed_env, gamma=GAMMA, learning_rate=LEARNING_RATE, epsilon=EPSILON, epsilon_decay = EPSILON_DECAY,
                            epsilon_min=EPSILON_MIN, learning_rate_decay=LEARNING_RATE_DECAY, t_max=100)
    rewards = []
    for i in range(2000):
        reward = agent.learn_from_episode()
        print("New reward: " + str(reward))
        rewards.append(reward)
    draw_rewards(rewards)


    policy = agent.policy()
    print_policy(policy)

else:
    # Definim combinacions de paràmetres
    param_grid = {
        "NUM_EPISODES": [500, 1000, 2000],
        "GAMMA": [0.90, 0.95, 0.99],
        "LEARNING_RATE": [0.1, 0.5, 0.8],
        "LEARNING_RATE_DECAY": [1.0, 0.999],
        "EPSILON": [1.0],
        "EPSILON_DECAY": [0.99, 0.995, 0.999],
        "EPSILON_MIN": [0.01]
    }

    keys, values = zip(*param_grid.items())
    experiments = [dict(zip(keys, v)) for v in product(*values)]

    results = []

    for i, config in enumerate(experiments):
        print(f"\nExperiment {i+1}/{len(experiments)}: {config}")
        env = gym.make("CliffWalking-v0", is_slippery=SLIPPERY)
        fixed_env = CustomCliffWrapper(env)

        agent = QLearningAgent(
            fixed_env,
            gamma=config["GAMMA"],
            learning_rate=config["LEARNING_RATE"],
            epsilon=config["EPSILON"],
            epsilon_decay=config["EPSILON_DECAY"],
            epsilon_min=config["EPSILON_MIN"],
            learning_rate_decay=config["LEARNING_RATE_DECAY"],
            t_max=T_MAX
        )

        start = time.time()
        episode_rewards = []
        for _ in range(config["NUM_EPISODES"]):
            episode_rewards.append(agent.learn_from_episode())
        end = time.time()

        mean_reward = np.mean(episode_rewards[-100:])  # recompenses finals
        policy = agent.policy()
        optimal_actions = np.sum(np.array(policy) == np.argmax(agent.Q, axis=1)) / len(policy)

        results.append({
            "config": config,
            "mean_last100_reward": mean_reward,
            "training_time": end - start,
            "optimality_ratio": optimal_actions
        })

    # Converteix a DataFrame i guarda
    df_results = pd.DataFrame(results)
    df_results.to_csv("experiments_Qlearning_results.csv", index=False)
    print("\n🔍 Experiments finalitzats. Resultats desats a 'experiments_results.csv'")

    visualitza()