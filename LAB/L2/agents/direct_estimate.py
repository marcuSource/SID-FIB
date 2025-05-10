import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from tqdm import tqdm
import random
import seaborn as sns
import os
from itertools import product

class DirectEstimationAgent:
    """
    Agent que implementa l'algoritme d'Estimació Directa per l'entorn CliffWalking.
    """
    
    def __init__(self, env, gamma=0.99, theta=1e-6, planning_steps=5, custom_reward=None):
        """
        Inicialitza l'agent d'Estimació Directa.
        
        Args:
            env: Entorn de Gym
            gamma: Factor de descompte per valors futurs (entre 0 i 1)
            theta: Llindar de convergència per aturar l'algoritme
            planning_steps: Nombre de passos de planificació per cada pas real
            custom_reward: Funció per modificar la recompensa original
        """
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.planning_steps = planning_steps
        self.custom_reward = custom_reward
        
        # Obtenir dimensions de l'entorn
        self.nS = env.observation_space.n  # Nombre d'estats
        self.nA = env.action_space.n       # Nombre d'accions
        
        # Inicialitzar el model
        self.initialize_model()
        
        # Inicialitzar la funció de valor i la política
        self.V = np.zeros(self.nS)
        self.Q = np.zeros((self.nS, self.nA))
        self.policy = np.zeros(self.nS, dtype=int)
        
        # Metriques
        self.rewards_history = []
        self.steps_history = []
    
    def initialize_model(self):
        """
        Inicialitza el model de l'agent.
        El model consisteix en matrius de transició i recompensa.
        """
        # Model de transició: P[s, a, s'] = prob. de transició d'estat s a s' amb acció a
        self.transition_counts = np.zeros((self.nS, self.nA, self.nS))
        self.transition_probs = np.ones((self.nS, self.nA, self.nS)) / self.nS
        
        # Model de recompensa: R[s, a, s'] = recompensa esperada en transició de s a s' amb acció a
        self.reward_sums = np.zeros((self.nS, self.nA, self.nS))
        self.reward_model = np.zeros((self.nS, self.nA, self.nS))
        
        # Conjunt d'estats terminals
        self.terminals = np.zeros(self.nS, dtype=bool)
        
        # Conjunt d'estats visitats
        self.visited_states = set()
    
    def update_model(self, state, action, next_state, reward, done):
        """
        Actualitza el model de l'entorn amb una nova transició.
        
        Args:
            state: Estat actual
            action: Acció realitzada
            next_state: Següent estat
            reward: Recompensa obtinguda
            done: Si l'episodi ha acabat
        """
        # Modifica la recompensa si s'ha especificat una funció personalitzada
        if self.custom_reward:
            reward = self.custom_reward(state, action, next_state, reward, done)
            
        # Actualitza comptes de transició
        self.transition_counts[state, action, next_state] += 1
        
        # Actualitza model de transició
        count_sa = np.sum(self.transition_counts[state, action])
        if count_sa > 0:
            self.transition_probs[state, action] = self.transition_counts[state, action] / count_sa
        
        # Actualitza model de recompensa
        self.reward_sums[state, action, next_state] += reward
        if self.transition_counts[state, action, next_state] > 0:
            self.reward_model[state, action, next_state] = (
                self.reward_sums[state, action, next_state] / 
                self.transition_counts[state, action, next_state]
            )
        
        # Actualitza conjunt d'estats terminals
        if done:
            self.terminals[next_state] = True
        
        # Actualitza conjunt d'estats visitats
        self.visited_states.add(state)
    
    def plan(self):
        """
        Realitza planificació (value iteration) utilitzant el model après.
        """
        # Planificació mitjançant iteració de valor
        for _ in range(self.planning_steps):
            delta = 0
            
            # Actualitzar valors per a estats visitats
            for s in self.visited_states:
                v = self.V[s]
                
                # Calcular valors d'acció
                action_values = np.zeros(self.nA)
                for a in range(self.nA):
                    for s_prime in range(self.nS):
                        # Probabilitat de transició
                        p = self.transition_probs[s, a, s_prime]
                        if p > 0:
                            # Recompensa esperada
                            r = self.reward_model[s, a, s_prime]
                            # Actualitzar valor d'acció
                            if self.terminals[s_prime]:
                                action_values[a] += p * r
                            else:
                                action_values[a] += p * (r + self.gamma * self.V[s_prime])
                
                # Actualitzar valor d'estat i funció Q
                best_action = np.argmax(action_values)
                self.V[s] = action_values[best_action]
                self.Q[s] = action_values
                self.policy[s] = best_action
                
                # Actualitzar delta
                delta = max(delta, abs(v - self.V[s]))
            
            # Si s'ha convergit, sortim
            if delta < self.theta:
                break
    
    def choose_action(self, state, epsilon=0.1):
        """
        Selecciona una acció utilitzant una política epsilon-greedy.
        
        Args:
            state: Estat actual
            epsilon: Probabilitat d'exploració
            
        Returns:
            int: Acció seleccionada
        """
        if random.random() < epsilon:
            # Exploració: acció aleatòria
            return random.randint(0, self.nA - 1)
        else:
            # Explotació: millor acció segons model
            return self.policy[state]
    
    def train(self, num_episodes=500, max_steps=100, epsilon_start=1.0, 
              epsilon_end=0.1, epsilon_decay=0.995, verbose=False):
        """
        Entrena l'agent mitjançant estimació directa.
        
        Args:
            num_episodes: Nombre d'episodis a entrenar
            max_steps: Màxim de passos per episodi
            epsilon_start: Valor inicial d'epsilon (exploració)
            epsilon_end: Valor mínim d'epsilon
            epsilon_decay: Factor de decaïment d'epsilon
            verbose: Si s'ha de mostrar informació durant l'entrenament
            
        Returns:
            dict: Resultats de l'entrenament
        """
        # Registrem temps d'inici
        start_time = time.time()
        
        # Historial d'entrenament
        rewards_history = []
        steps_history = []
        epsilon_history = []
        epsilon = epsilon_start
        
        if verbose:
            print(f"Iniciant Estimació Directa amb gamma={self.gamma}, planning_steps={self.planning_steps}")
            print(f"Entorn: {self.nS} estats, {self.nA} accions")
            iterator = tqdm(range(num_episodes))
        else:
            iterator = range(num_episodes)
        
        for episode in iterator:
            # Reiniciar l'entorn
            state, _ = self.env.reset()
            episode_reward = 0
            steps = 0
            done = False
            
            while not done and steps < max_steps:
                # Seleccionar acció
                action = self.choose_action(state, epsilon)
                
                # Executar acció
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Actualitzar estadístiques
                episode_reward += reward
                steps += 1
                
                # Actualitzar model
                self.update_model(state, action, next_state, reward, done)
                
                # Planificar (actualitzar valors i política)
                self.plan()
                
                # Actualitzar estat
                state = next_state
            
            # Actualitzar epsilon
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            
            # Registrar resultats
            rewards_history.append(episode_reward)
            steps_history.append(steps)
            epsilon_history.append(epsilon)
            
            # Mostrar progressió si verbose
            if verbose and (episode + 1) % 50 == 0:
                avg_reward = np.mean(rewards_history[-50:])
                avg_steps = np.mean(steps_history[-50:])
                print(f"Episodi {episode+1}: Recompensa={avg_reward:.2f}, Passos={avg_steps:.2f}, Epsilon={epsilon:.4f}")
        
        # Registrem temps total
        total_time = time.time() - start_time
        
        # Guardem l'historial
        self.rewards_history = rewards_history
        self.steps_history = steps_history
        
        if verbose:
            print(f"Entrenament completat en {total_time:.4f} segons")
            print(f"Recompensa final: {np.mean(rewards_history[-10:]):.2f}")
        
        return {
            "episodes": num_episodes,
            "rewards_history": rewards_history,
            "steps_history": steps_history,
            "epsilon_history": epsilon_history,
            "total_time": total_time,
            "mean_last100_reward": np.mean(rewards_history[-100:]),
            "std_last100_reward": np.std(rewards_history[-100:])
        }
    
    def get_action(self, state):
        """
        Retorna l'acció òptima per a un estat segons la política apresa.
        
        Args:
            state: Estat actual
            
        Returns:
            int: Acció òptima
        """
        return self.policy[state]
    
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
    
    def visualize_value_function(self, save_path=None):
        """
        Visualitza la funció de valor apresa.
        
        Args:
            save_path: Ruta on es guardarà la imatge (opcional)
        """
        # CliffWalking té una forma de 4x12
        V_grid = np.zeros((4, 12))
        for s in range(self.nS):
            i, j = s // 12, s % 12
            V_grid[i, j] = self.V[s]
        
        plt.figure(figsize=(12, 5))
        plt.imshow(V_grid, cmap='viridis')
        plt.colorbar(label='Valor')
        plt.title(f'Funció de Valor - Estimació Directa (gamma={self.gamma})')
        
        # Marcar el precipici
        for j in range(1, 11):
            plt.text(j, 3, 'C', ha='center', va='center', color='white')
        
        # Marcar inici i objectiu
        plt.text(0, 3, 'S', ha='center', va='center', color='white', fontsize=12)
        plt.text(11, 3, 'G', ha='center', va='center', color='white', fontsize=12)
        
        plt.tight_layout()
        
        # Guardar la imatge si s'ha especificat la ruta
        if save_path:
            # Crear el directori si no existeix
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"Funció de valor guardada a: {save_path}")
        
        plt.show()
    
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
            a = self.policy[s]
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
        plt.title(f'Política - Estimació Directa (gamma={self.gamma})')
        plt.tight_layout()
        
        # Guardar la imatge si s'ha especificat la ruta
        if save_path:
            # Crear el directori si no existeix
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"Política guardada a: {save_path}")
        
        plt.show()
    
    def plot_training_progress(self, window_size=10, save_path=None):
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
def run_experiments(output_dir="experiments_direct_estimation"):
    # Crear directori per guardar resultats
    os.makedirs(output_dir, exist_ok=True)
    
    # Paràmetres a experimentar
    gamma_values = [0.9, 0.95, 0.99, 0.999]
    planning_steps_values = [1, 3, 5, 10]
    epsilon_decay_values = [0.99, 0.995, 0.999]
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
    base_planning_steps = 5
    base_epsilon_decay = 0.995
    base_reward_function = "default"
    
    # Experiments variant gamma
    experiments_gamma = [(gamma, base_planning_steps, base_epsilon_decay, base_reward_function) 
                         for gamma in gamma_values]
    
    # Experiments variant planning_steps
    experiments_planning = [(base_gamma, steps, base_epsilon_decay, base_reward_function) 
                           for steps in planning_steps_values]
    
    # Experiments variant epsilon_decay
    experiments_epsilon = [(base_gamma, base_planning_steps, decay, base_reward_function) 
                          for decay in epsilon_decay_values]
    
    # Experiments variant reward_function
    experiments_reward = [(base_gamma, base_planning_steps, base_epsilon_decay, reward_name) 
                         for reward_name in reward_functions.keys()]
    
    # Combinar tots els experiments
    all_experiments = (experiments_gamma + experiments_planning + 
                      experiments_epsilon + experiments_reward)
    
    # Eliminar duplicats
    all_experiments = list(set(all_experiments))
    
    total_experiments = len(all_experiments)
    print(f"Iniciant {total_experiments} experiments...")
    
    # Executar experiments
    for i, (gamma, planning_steps, epsilon_decay, reward_name) in enumerate(all_experiments):
        print(f"Experiment {i+1}/{total_experiments}: gamma={gamma}, planning_steps={planning_steps}, "
              f"epsilon_decay={epsilon_decay}, reward={reward_name}")
        
        # Crear entorn
        env = gym.make('CliffWalking-v0', is_slippery=True)
        
        # Obtenir funció de recompensa
        reward_func = reward_functions[reward_name]
        
        # Crear i entrenar agent
        agent = DirectEstimationAgent(env, gamma=gamma, planning_steps=planning_steps, 
                                      custom_reward=reward_func)
        
        train_results = agent.train(num_episodes=500, epsilon_decay=epsilon_decay, verbose=False)
        
        # Avaluar agent
        print(f"Avaluant agent amb 500 episodis...")
        eval_results = agent.evaluate(num_episodes=500)
        
        # Guardar resultat
        results.append({
            "gamma": gamma,
            "planning_steps": planning_steps,
            "epsilon_decay": epsilon_decay,
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
        if (gamma == base_gamma and planning_steps == base_planning_steps and 
            epsilon_decay == base_epsilon_decay and reward_name == base_reward_function):
            
            # Guardar visualitzacions
            agent.visualize_value_function(save_path=os.path.join(output_dir, "valor_funcio_base.png"))
            agent.visualize_policy(save_path=os.path.join(output_dir, "politica_base.png"))
            agent.plot_training_progress(save_path=os.path.join(output_dir, "progres_entrenament_base.png"))
        
        # Tancar entorn
        env.close()
    
    # Convertir a DataFrame
    results_df = pd.DataFrame(results)
    
    # Guardar a CSV
    csv_path = os.path.join(output_dir, "direct_estimation_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Resultats guardats a '{csv_path}'")
    
    return results_df

# Funció per visualitzar resultats
def visualize_results(results_df=None, output_dir="experiments_direct_estimation"):
    # Crear directori per guardar resultats
    os.makedirs(output_dir, exist_ok=True)
    
    if results_df is None:
        try:
            csv_path = os.path.join(output_dir, "direct_estimation_results.csv")
            results_df = pd.read_csv(csv_path)
        except FileNotFoundError:
            print(f"No s'ha trobat el fitxer de resultats a {csv_path}. Executa primer els experiments.")
            return
    
    # Configurar estil
    sns.set(style="whitegrid")
    
    # Figura 1: Efecte de gamma en la recompensa mitjana per diferents funcions de recompensa
    plt.figure(figsize=(10, 6))
    gamma_df = results_df[results_df['reward_function'] == 'default']
    sns.lineplot(data=gamma_df, x="gamma", y="eval_mean_reward", marker="o")
    plt.title("Efecte de gamma en la recompensa mitjana d'avaluació")
    plt.xlabel("Gamma")
    plt.ylabel("Recompensa mitjana")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "de_gamma_vs_reward.png"))
    plt.show()
    
    # Figura 2: Efecte de planning_steps en la recompensa
    plt.figure(figsize=(10, 6))
    planning_df = results_df[results_df['planning_steps'].isin([1, 3, 5, 10])]
    sns.lineplot(data=planning_df, x="planning_steps", y="eval_mean_reward", marker="o")
    plt.title("Efecte del nombre de passos de planificació en la recompensa")
    plt.xlabel("Passos de planificació")
    plt.ylabel("Recompensa mitjana")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "de_planning_vs_reward.png"))
    plt.show()
    
    # Figura 3: Efecte d'epsilon_decay en la recompensa
    plt.figure(figsize=(10, 6))
    epsilon_df = results_df[results_df['epsilon_decay'].isin([0.99, 0.995, 0.999])]
    sns.lineplot(data=epsilon_df, x="epsilon_decay", y="eval_mean_reward", marker="o")
    plt.title("Efecte del decaïment d'epsilon en la recompensa")
    plt.xlabel("Epsilon Decay")
    plt.ylabel("Recompensa mitjana")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "de_epsilon_vs_reward.png"))
    plt.show()
    
    # Figura 4: Comparació de diferents funcions de recompensa
    plt.figure(figsize=(10, 6))
    reward_df = results_df.drop_duplicates(subset=['reward_function'])
    sns.barplot(data=reward_df, x="reward_function", y="eval_mean_reward")
    plt.title("Comparació de diferents funcions de recompensa")
    plt.xlabel("Funció de recompensa")
    plt.ylabel("Recompensa mitjana")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "de_reward_function_comparison.png"))
    plt.show()
    
    # Figura 5: Taxa d'èxit segons diferents paràmetres
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=results_df, x="gamma", y="success_rate", 
                   hue="reward_function", size="planning_steps", sizes=(50, 200))
    plt.title("Taxa d'èxit segons diferents paràmetres")
    plt.xlabel("Gamma")
    plt.ylabel("Taxa d'èxit")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "de_success_rate.png"))
    plt.show()
    
    # Figura 6: Temps d'entrenament vs Planning Steps
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=planning_df, x="planning_steps", y="training_time", marker="o")
    plt.title("Temps d'entrenament vs Passos de planificació")
    plt.xlabel("Passos de planificació")
    plt.ylabel("Temps d'entrenament (s)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "de_planning_vs_time.png"))
    plt.show()
    
    # Figura 7: Distribució del nombre de passos segons la funció de recompensa
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=reward_df, x="reward_function", y="eval_mean_steps")
    plt.title("Distribució del nombre de passos segons la funció de recompensa")
    plt.xlabel("Funció de recompensa")
    plt.ylabel("Nombre mitjà de passos")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "de_reward_steps.png"))
    plt.show()

# Funció principal
def main():
    # Crear directori per guardar resultats
    experiments_dir = "experiments_direct_estimation"
    os.makedirs(experiments_dir, exist_ok=True)
    
    # Escollir mode
    print("1. Executar experiments complets")
    print("2. Visualitzar resultats existents")
    print("3. Executar un únic experiment amb configuració base")
    mode = input("Escull una opció (1-3): ")
    
    if mode == "1":
        results_df = run_experiments(output_dir=experiments_dir)
        visualize_results(results_df, output_dir=experiments_dir)
    elif mode == "2":
        visualize_results(output_dir=experiments_dir)
    elif mode == "3":
        run_single_experiment(output_dir=experiments_dir)
    else:
        print("Opció no vàlida")

def run_single_experiment(gamma=0.99, planning_steps=5, epsilon_decay=0.995, output_dir="experiments_direct_estimation"):
    """
    Executa un únic experiment d'Estimació Directa amb la configuració base.
    
    Args:
        gamma: Factor de descompte
        planning_steps: Nombre de passos de planificació
        epsilon_decay: Factor de decaïment d'epsilon
        output_dir: Directori on guardar els resultats
    """
    # Crear directori per guardar resultats
    os.makedirs(output_dir, exist_ok=True)
    
    # Crear l'entorn
    env = gym.make('CliffWalking-v0', is_slippery=True)
    
    # Crear i entrenar agent
    print(f"Executant experiment amb gamma={gamma}, planning_steps={planning_steps}, epsilon_decay={epsilon_decay}")
    agent = DirectEstimationAgent(env, gamma=gamma, planning_steps=planning_steps)
    
    # Entrenar agent
    train_results = agent.train(num_episodes=500, epsilon_decay=epsilon_decay, verbose=True)
    
    # Visualitzar progressió d'entrenament
    agent.plot_training_progress(save_path=os.path.join(output_dir, "progres_entrenament.png"))
    
    # Visualitzar funció de valor i política
    agent.visualize_value_function(save_path=os.path.join(output_dir, "valor_funcio.png"))
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

if __name__ == "__main__":
    main()