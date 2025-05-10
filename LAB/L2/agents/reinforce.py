import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import os
from itertools import product
from collections import deque
import random

class REINFORCEAgent:
    """
    Agent que implementa l'algoritme REINFORCE per l'entorn CliffWalking utilitzant només NumPy.
    """
    
    def __init__(self, env, gamma=0.99, learning_rate=0.001, custom_reward=None):
        """
        Inicialitza l'agent REINFORCE.
        
        Args:
            env: Entorn de Gym
            gamma: Factor de descompte per valors futurs (entre 0 i 1)
            learning_rate: Taxa d'aprenentatge per a l'optimitzador
            custom_reward: Funció per modificar la recompensa original
        """
        self.env = env
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.custom_reward = custom_reward
        
        # Obtenir dimensions de l'entorn
        self.nS = env.observation_space.n  # Nombre d'estats
        self.nA = env.action_space.n       # Nombre d'accions
        
        # Inicialitzar paràmetres de la política (model lineal senzill)
        self.weights = np.random.randn(self.nS, self.nA) * 0.01
        
        # Historial d'entrenament
        self.rewards_history = []
        self.steps_history = []
    
    def get_policy(self, state):
        """
        Calcula les probabilitats d'acció per a un estat donat.
        
        Args:
            state: Índex d'estat actual
            
        Returns:
            array: Probabilitats per a cada acció
        """
        # Convertir l'índex d'estat a vector one-hot
        state_one_hot = np.zeros(self.nS)
        state_one_hot[state] = 1.0
        
        # Calcular puntuacions per a cada acció
        action_scores = np.dot(state_one_hot, self.weights)
        
        # Convertir puntuacions a probabilitats (softmax)
        exp_scores = np.exp(action_scores - np.max(action_scores))  # Restar màxim per estabilitat numèrica
        action_probs = exp_scores / np.sum(exp_scores)
        
        return action_probs
    
    def choose_action(self, state):
        """
        Selecciona una acció segons la política actual.
        
        Args:
            state: Índex d'estat actual
            
        Returns:
            int: Acció seleccionada
        """
        # Obtenir probabilitats d'acció
        action_probs = self.get_policy(state)
        
        # Mostrejar acció segons distribució de probabilitat
        action = np.random.choice(self.nA, p=action_probs)
        
        return action
    
    def train_episode(self, max_steps=100):
        """
        Entrena un episodi complet utilitzant REINFORCE.
        
        Args:
            max_steps: Màxim de passos per episodi
            
        Returns:
            float: Recompensa total de l'episodi
            int: Nombre de passos en l'episodi
        """
        # Reiniciar entorn
        state, _ = self.env.reset()
        
        # Llistes per guardar trajectòria
        states = []
        actions = []
        rewards = []
        
        # Executar episodi
        episode_reward = 0
        steps = 0
        done = False
        
        while not done and steps < max_steps:
            # Seleccionar acció
            action = self.choose_action(state)
            
            # Executar acció
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            # Aplicar funció de recompensa personalitzada si s'ha especificat
            if self.custom_reward is not None:
                reward = self.custom_reward(state, action, next_state, reward, done)
            
            # Guardar transició
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            # Actualitzar estadístiques
            episode_reward += reward
            steps += 1
            state = next_state
        
        # Calcular retorns descomptats
        returns = self._calculate_returns(rewards)
        
        # Actualitzar política
        self._update_policy(states, actions, returns)
        
        return episode_reward, steps
    
    def _calculate_returns(self, rewards):
        """
        Calcula els retorns descomptats per a cada pas de l'episodi.
        
        Args:
            rewards: Llista de recompenses de l'episodi
            
        Returns:
            Llista de retorns descomptats
        """
        returns = []
        G = 0
        
        # Calcular el retorn total a partir de l'últim pas de l'episodi
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        
        # Normalitzar retorns per reduir variància
        returns = np.array(returns)
        if len(returns) > 1:
            returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-9)
        
        return returns
    
    def _update_policy(self, states, actions, returns):
        """
        Actualitza els paràmetres de la política usant gradients REINFORCE.
        
        Args:
            states: Llista d'estats de l'episodi
            actions: Llista d'accions de l'episodi
            returns: Llista de retorns descomptats
        """
        for t in range(len(states)):
            state = states[t]
            action = actions[t]
            G = returns[t]
            
            # Obtenir probabilitats d'acció per a l'estat actual
            action_probs = self.get_policy(state)
            
            # Actualitzar els pesos per a totes les accions
            for a in range(self.nA):
                # Construir el terme del gradient
                if a == action:
                    # Per l'acció presa: log(π(a|s)) * G
                    gradient = G * (1 - action_probs[a])
                else:
                    # Per la resta d'accions: -log(π(a|s)) * G
                    gradient = -G * action_probs[a]
                
                # Actualitzar pesos
                self.weights[state, a] += self.learning_rate * gradient
    
    def train(self, num_episodes=3000, max_steps=100, verbose=False):
        """
        Entrena l'agent utilitzant l'algoritme REINFORCE.
        
        Args:
            num_episodes: Nombre d'episodis a entrenar (augmentat a 3000)
            max_steps: Màxim de passos per episodi
            verbose: Si s'ha de mostrar informació durant l'entrenament
            
        Returns:
            dict: Resultats de l'entrenament
        """
        # Registrem temps d'inici
        start_time = time.time()
        
        # Historial d'entrenament
        rewards_history = []
        steps_history = []
        
        if verbose:
            print(f"Iniciant REINFORCE amb gamma={self.gamma}, lr={self.learning_rate}")
            print(f"Entorn: {self.nS} estats, {self.nA} accions")
            print(f"Entrenament amb {num_episodes} episodis")
            iterator = tqdm(range(num_episodes))
        else:
            iterator = range(num_episodes)
        
        for episode in iterator:
            # Entrenar un episodi
            episode_reward, steps = self.train_episode(max_steps)
            
            # Registrar resultats
            rewards_history.append(episode_reward)
            steps_history.append(steps)
            
            # Mostrar progressió si verbose
            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(rewards_history[-100:])
                avg_steps = np.mean(steps_history[-100:])
                print(f"Episodi {episode+1}: Recompensa={avg_reward:.2f}, Passos={avg_steps:.2f}")
        
        # Registrem temps total
        total_time = time.time() - start_time
        
        # Guardem l'historial
        self.rewards_history = rewards_history
        self.steps_history = steps_history
        
        if verbose:
            print(f"Entrenament completat en {total_time:.4f} segons")
            print(f"Recompensa final: {np.mean(rewards_history[-100:]):.2f}")
        
        return {
            "episodes": num_episodes,
            "rewards_history": rewards_history,
            "steps_history": steps_history,
            "total_time": total_time,
            "mean_last100_reward": np.mean(rewards_history[-100:]),
            "std_last100_reward": np.std(rewards_history[-100:])
        }
    
    def get_action(self, state):
        """
        Retorna l'acció amb màxima probabilitat per a un estat segons la política apresa.
        
        Args:
            state: Estat actual
            
        Returns:
            int: Acció òptima
        """
        # Obtenir probabilitats d'acció
        action_probs = self.get_policy(state)
        
        # Seleccionar acció amb màxima probabilitat
        return np.argmax(action_probs)
    
    def evaluate(self, num_episodes=500, max_steps=100, render=False):
        """
        Avalua la política apresa.
        
        Args:
            num_episodes: Nombre d'episodis a avaluar
            max_steps: Màxim de passos per episodi
            render: Si s'ha de visualitzar l'entorn
            
        Returns:
            dict: Resultats de l'avaluació
        """
        total_rewards = []
        total_steps = []
        success_rate = 0
        
        eval_env = gym.make('CliffWalking-v0', is_slippery=True)
        
        for episode in range(num_episodes):
            state, _ = eval_env.reset()
            episode_reward = 0
            steps = 0
            done = False
            success = False
            
            while not done and steps < max_steps:
                # Seleccionar acció segons la política
                action = self.get_action(state)
                
                # Executar acció
                next_state, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                
                # Comprovar si ha arribat a l'objectiu
                if terminated and reward == 0:  # En CliffWalking, l'objectiu dona recompensa 0
                    success = True
                
                # Actualitzar estadístiques
                episode_reward += reward
                steps += 1
                state = next_state
                
                # Visualitzar si es demana
                if render:
                    eval_env.render()
                    time.sleep(0.1)
            
            if success:
                success_rate += 1
            
            total_rewards.append(episode_reward)
            total_steps.append(steps)
        
        eval_env.close()
        
        success_rate = success_rate / num_episodes
        
        return {
            "mean_reward": np.mean(total_rewards),
            "std_reward": np.std(total_rewards),
            "mean_steps": np.mean(total_steps),
            "std_steps": np.std(total_steps),
            "rewards": total_rewards,
            "steps": total_steps,
            "success_rate": success_rate
        }
    
    def visualize_policy(self, save_path=None):
        """
        Visualitza la política apresa.
        
        Args:
            save_path: Ruta on es guardarà la imatge (opcional)
        """
        # CliffWalking té una forma de 4x12
        arrows = ['↑', '→', '↓', '←']
        policy_grid = np.empty((4, 12), dtype=object)
        
        for s in range(self.nS):
            i, j = s // 12, s % 12
            a = self.get_action(s)
            policy_grid[i, j] = arrows[a]
        
        # El precipici no té política
        for j in range(1, 11):
            policy_grid[3, j] = 'C'
        
        # Marcar inici i objectiu
        policy_grid[3, 0] = 'S'
        policy_grid[3, 11] = 'G'
        
        plt.figure(figsize=(12, 5))
        for i in range(4):
            for j in range(12):
                plt.text(j, i, policy_grid[i, j], ha='center', va='center', fontsize=12)
                
        plt.grid(True)
        plt.xlim(-0.5, 11.5)
        plt.ylim(3.5, -0.5)
        plt.title(f'Política - REINFORCE (gamma={self.gamma}, 3000 episodis)')
        plt.tight_layout()
        
        # Guardar la imatge si s'ha especificat la ruta
        if save_path:
            # Crear el directori si no existeix
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"Política guardada a: {save_path}")
        
        plt.show()
    
    def plot_training_progress(self, window_size=50, save_path=None):
        """
        Visualitza la progressió de l'entrenament.
        
        Args:
            window_size: Mida de finestra per suavitzar les corbes
            save_path: Ruta on es guardarà la imatge (opcional)
        """
        rewards = self.rewards_history
        steps = self.steps_history
        
        # Suavitzem les corbes amb una mitjana mòbil
        def moving_average(data, window_size):
            return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
        
        smoothed_rewards = moving_average(rewards, window_size) if len(rewards) >= window_size else rewards
        smoothed_steps = moving_average(steps, window_size) if len(steps) >= window_size else steps
        
        plt.figure(figsize=(12, 5))
        
        # Gràfic de recompenses
        plt.subplot(1, 2, 1)
        plt.plot(rewards, 'b-', alpha=0.3)
        if len(rewards) >= window_size:
            plt.plot(np.arange(window_size-1, len(rewards)), smoothed_rewards, 'b-')
        plt.title('Recompensa per episodi')
        plt.xlabel('Episodi')
        plt.ylabel('Recompensa')
        plt.grid(True)
        
        # Gràfic de passos
        plt.subplot(1, 2, 2)
        plt.plot(steps, 'r-', alpha=0.3)
        if len(steps) >= window_size:
            plt.plot(np.arange(window_size-1, len(steps)), smoothed_steps, 'r-')
        plt.title('Passos per episodi')
        plt.xlabel('Episodi')
        plt.ylabel('Passos')
        plt.grid(True)
        
        plt.tight_layout()
        
        # Guardar la imatge si s'ha especificat la ruta
        if save_path:
            # Crear el directori si no existeix
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"Progrés d'entrenament guardat a: {save_path}")
        
        plt.show()

# Definim funcions de recompensa personalitzades
def default_reward(s, a, next_s, reward, done):
    """Retorna la recompensa per defecte sense modificacions"""
    return reward

def step_penalty_reward(s, a, next_s, reward, done):
    """Afegeix una petita penalització per cada pas per fomentar camins més curts"""
    # Penalització per cada pas
    new_reward = reward - 0.01
    
    # Bonus per arribar a l'objectiu
    if done and reward == 0:  # En CliffWalking, arribar a l'objectiu dona recompensa 0
        new_reward += 1.0
    
    return new_reward

def cliff_avoidance_reward(s, a, next_s, reward, done):
    """Augmenta la penalització per caure pel precipici"""
    if reward == -100:  # La recompensa del precipici en CliffWalking
        return -200  # Augmentem la penalització
    return reward

# Funció per executar experiments i guardar resultats
def run_experiments(output_dir="experiments_reinforce"):
    # Crear directori per guardar resultats
    os.makedirs(output_dir, exist_ok=True)
    
    # Paràmetres a experimentar
    gamma_values = [0.9, 0.95, 0.99, 0.999]
    learning_rate_values = [0.001, 0.01, 0.05]  # Afegit un valor més alt (0.05)
    reward_functions = {
        "default": default_reward,
        "step_penalty": step_penalty_reward,
        "cliff_avoidance": cliff_avoidance_reward
    }
    
    # Preparar DataFrame per resultats
    results = []
    
    # Generar combinacions de paràmetres
    # Per limitar el nombre total d'experiments, només variarem un paràmetre a la vegada des de la configuració base
    base_gamma = 0.99
    base_learning_rate = 0.01  # Canviat a 0.01 segons els resultats anteriors
    base_reward_function = "default"  # Canviat a step_penalty segons els resultats anteriors
    
    # Experiments variant gamma
    experiments_gamma = [(gamma, base_learning_rate, base_reward_function) 
                         for gamma in gamma_values]
    
    # Experiments variant learning_rate
    experiments_lr = [(base_gamma, lr, base_reward_function) 
                      for lr in learning_rate_values]
    
    # Experiments variant reward_function
    experiments_reward = [(base_gamma, base_learning_rate, reward_name) 
                         for reward_name in reward_functions.keys()]
    
    # Combinar tots els experiments
    all_experiments = (experiments_gamma + experiments_lr + experiments_reward)
    
    # Eliminar duplicats
    all_experiments = list(set(all_experiments))
    
    total_experiments = len(all_experiments)
    print(f"Iniciant {total_experiments} experiments amb 3000 episodis cadascun...")
    
    # Executar experiments
    for i, (gamma, lr, reward_name) in enumerate(all_experiments):
        print(f"Experiment {i+1}/{total_experiments}: gamma={gamma}, lr={lr}, "
              f"reward={reward_name}")
        
        # Crear entorn
        env = gym.make('CliffWalking-v0', is_slippery=True)
        
        # Obtenir funció de recompensa
        reward_func = reward_functions[reward_name]
        
        # Crear i entrenar agent
        agent = REINFORCEAgent(
            env, 
            gamma=gamma, 
            learning_rate=lr,
            custom_reward=reward_func
        )
        
        # Entrenar amb 3000 episodis
        train_results = agent.train(num_episodes=3000, verbose=True)
        
        # Avaluar agent
        print(f"Avaluant agent amb 500 episodis...")
        eval_results = agent.evaluate(num_episodes=500)
        
        # Guardar resultat
        results.append({
            "gamma": gamma,
            "learning_rate": lr,
            "reward_function": reward_name,
            "training_time": train_results["total_time"],
            "mean_last100_reward": train_results["mean_last100_reward"],
            "std_last100_reward": train_results["std_last100_reward"],
            "eval_mean_reward": eval_results["mean_reward"],
            "eval_std_reward": eval_results["std_reward"],
            "eval_mean_steps": eval_results["mean_steps"],
            "eval_std_steps": eval_results["std_steps"],
            "success_rate": eval_results["success_rate"]
        })
        
        # Guardar gràfics específics per la configuració base
        if (gamma == base_gamma and lr == base_learning_rate and reward_name == base_reward_function):
            
            # Guardar visualitzacions
            agent.visualize_policy(save_path=os.path.join(output_dir, "politica_base.png"))
            agent.plot_training_progress(save_path=os.path.join(output_dir, "progres_entrenament_base.png"))
        
        # Tancar entorn
        env.close()
    
    # Convertir a DataFrame
    results_df = pd.DataFrame(results)
    
    # Guardar a CSV
    csv_path = os.path.join(output_dir, "reinforce_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Resultats guardats a '{csv_path}'")
    
    return results_df

# Funció per visualitzar resultats
def visualize_results(results_df=None, output_dir="experiments_reinforce"):
    # Crear directori per guardar resultats
    os.makedirs(output_dir, exist_ok=True)
    
    if results_df is None:
        try:
            csv_path = os.path.join(output_dir, "reinforce_results.csv")
            results_df = pd.read_csv(csv_path)
        except FileNotFoundError:
            print(f"No s'ha trobat el fitxer de resultats a {csv_path}. Executa primer els experiments.")
            return
    
    # Configurar estil
    sns.set(style="whitegrid")
    
    # Figura 1: Efecte de gamma en la recompensa mitjana
    plt.figure(figsize=(10, 6))
    gamma_df = results_df[results_df['reward_function'] == 'default']
    sns.lineplot(data=gamma_df, x="gamma", y="eval_mean_reward", marker="o")
    plt.title("Efecte de gamma en la recompensa mitjana d'avaluació (3000 episodis)")
    plt.xlabel("Gamma")
    plt.ylabel("Recompensa mitjana")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "reinforce_gamma_vs_reward.png"))
    plt.show()
    
    # Figura 2: Efecte de learning_rate en la recompensa
    plt.figure(figsize=(10, 6))
    lr_df = results_df[results_df['reward_function'] == 'default']
    sns.lineplot(data=lr_df, x="learning_rate", y="eval_mean_reward", marker="o")
    plt.title("Efecte de la taxa d'aprenentatge en la recompensa (3000 episodis)")
    plt.xlabel("Taxa d'aprenentatge")
    plt.ylabel("Recompensa mitjana")
    plt.xscale('log')  # Escala logarítmica per la taxa d'aprenentatge
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "reinforce_lr_vs_reward.png"))
    plt.show()
    
    # Figura 3: Comparació de diferents funcions de recompensa
    plt.figure(figsize=(10, 6))
    reward_df = results_df.drop_duplicates(subset=['reward_function'])
    sns.barplot(data=reward_df, x="reward_function", y="eval_mean_reward")
    plt.title("Comparació de diferents funcions de recompensa (3000 episodis)")
    plt.xlabel("Funció de recompensa")
    plt.ylabel("Recompensa mitjana")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "reinforce_reward_function_comparison.png"))
    plt.show()
    
    # Figura 4: Taxa d'èxit segons gamma i funció de recompensa
    plt.figure(figsize=(10, 6))
    success_df = results_df[(results_df['gamma'].isin([0.9, 0.95, 0.99, 0.999])) & 
                           (results_df['reward_function'].isin(['default', 'step_penalty', 'cliff_avoidance']))]
    sns.barplot(data=success_df, x="gamma", y="success_rate", hue="reward_function")
    plt.title("Taxa d'èxit segons gamma i funció de recompensa (3000 episodis)")
    plt.xlabel("Gamma")
    plt.ylabel("Taxa d'èxit")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "reinforce_success_rate.png"))
    plt.show()
    
    # Figura 5: Temps d'entrenament segons taxa d'aprenentatge
    plt.figure(figsize=(10, 6))
    time_df = results_df[results_df['reward_function'] == 'default']
    sns.lineplot(data=time_df, x="learning_rate", y="training_time", marker="o")
    plt.title("Temps d'entrenament vs Taxa d'aprenentatge (3000 episodis)")
    plt.xlabel("Taxa d'aprenentatge")
    plt.ylabel("Temps d'entrenament (s)")
    plt.xscale('log')  # Escala logarítmica per la taxa d'aprenentatge
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "reinforce_lr_vs_time.png"))
    plt.show()
    
    # Figura 6: Distribució del nombre de passos segons la funció de recompensa
    plt.figure(figsize=(10, 6))
    steps_df = results_df.drop_duplicates(subset=['reward_function'])
    sns.boxplot(data=steps_df, x="reward_function", y="eval_mean_steps")
    plt.title("Distribució del nombre de passos segons la funció de recompensa (3000 episodis)")
    plt.xlabel("Funció de recompensa")
    plt.ylabel("Nombre mitjà de passos")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "reinforce_reward_steps.png"))
    plt.show()

# Funció per executar un únic experiment
def run_single_experiment(gamma=0.99, learning_rate=0.01, num_episodes=3000, 
                          output_dir="experiments_reinforce"):
    """
    Executa un únic experiment REINFORCE amb paràmetres específics.
    
    Args:
        gamma: Factor de descompte
        learning_rate: Taxa d'aprenentatge
        num_episodes: Nombre d'episodis a entrenar
        output_dir: Directori on guardar els resultats
    """
    # Crear directori per guardar resultats
    os.makedirs(output_dir, exist_ok=True)
    
    # Crear l'entorn
    env = gym.make('CliffWalking-v0', is_slippery=True)
    
    # Crear i entrenar agent
    print(f"Executant experiment amb gamma={gamma}, lr={learning_rate}, episodis={num_episodes}")
    agent = REINFORCEAgent(env, gamma=gamma, learning_rate=learning_rate)
    
    # Entrenar agent
    train_results = agent.train(num_episodes=num_episodes, verbose=True)
    
    # Visualitzar progressió d'entrenament
    agent.plot_training_progress(save_path=os.path.join(output_dir, "progres_entrenament.png"))
    
    # Visualitzar política
    agent.visualize_policy(save_path=os.path.join(output_dir, "politica.png"))
    
    # Avaluar agent
    print("\nAvaluant la política...")
    eval_results = agent.evaluate(num_episodes=500)
    
    print("Resultats de l'avaluació:")
    print(f"Recompensa mitjana: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"Passos mitjans: {eval_results['mean_steps']:.2f} ± {eval_results['std_steps']:.2f}")
    print(f"Taxa d'èxit: {eval_results['success_rate'] * 100:.2f}%")
    
    # Tancar entorn
    env.close()
    
    return agent, train_results, eval_results

# Funció principal
def main():
    # Crear directori per guardar resultats
    experiments_dir = "experiments_reinforce"
    os.makedirs(experiments_dir, exist_ok=True)
    
    # Escollir mode
    print("1. Executar experiments complets (3000 episodis)")
    print("2. Visualitzar resultats existents")
    print("3. Executar un únic experiment amb configuració base (3000 episodis)")
    mode = input("Escull una opció (1-3): ")
    
    if mode == "1":
        results_df = run_experiments(output_dir=experiments_dir)
        visualize_results(results_df, output_dir=experiments_dir)
    elif mode == "2":
        visualize_results(output_dir=experiments_dir)
    elif mode == "3":
        run_single_experiment(
            gamma=0.99, 
            learning_rate=0.01,  # Augmentat a 0.01 segons els resultats anteriors
            num_episodes=3000,
            output_dir=experiments_dir
        )
    else:
        print("Opció no vàlida")

if __name__ == "__main__":
    main()