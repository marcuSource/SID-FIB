import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from tqdm import tqdm
from itertools import product
import seaborn as sns
import os

class ValueIterationAgent:
    """
    Agent que implementa l'algorisme d'Iteració de Valor per l'entorn CliffWalking.
    """
    
    def __init__(self, env, gamma=0.99, theta=1e-6, custom_reward=None):
        """
        Inicialitza l'agent d'Iteració de Valor.
        
        Args:
            env: Entorn de Gym
            gamma: Factor de descompte per valors futurs (entre 0 i 1)
            theta: Llindar de convergència per aturar l'algorisme
            custom_reward: Funció per modificar la recompensa original
        """
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.custom_reward = custom_reward
        
        # Obtenir dimensions de l'entorn
        self.nS = env.observation_space.n  # Nombre d'estats (48 per CliffWalking)
        self.nA = env.action_space.n       # Nombre d'accions (4 per CliffWalking)
        
        # Inicialitzar la funció de valor d'estat
        self.V = np.zeros(self.nS)
        
        # Inicialitzar la funció Q i la política
        self.Q = np.zeros((self.nS, self.nA))
        self.policy = np.zeros(self.nS, dtype=int)
        
        # Obtenir el model de transició de l'entorn
        self.P = self._get_model()
        
        # Aplicar modificació de recompenses si s'ha especificat
        if self.custom_reward:
            self._apply_custom_reward()
    
    def _get_model(self):
        """
        Obté el model de transició de l'entorn.
        
        Returns:
            dict: Model de transició P[estat][acció] = [(prob, següent_estat, recompensa, terminal)]
        """
        return self.env.unwrapped.P
    
    def _apply_custom_reward(self):
        """
        Modifica les recompenses del model segons la funció custom_reward.
        """
        for s in range(self.nS):
            for a in range(self.nA):
                for i, (prob, next_s, reward, done) in enumerate(self.P[s][a]):
                    # Substituïm la recompensa original per la personalitzada
                    new_reward = self.custom_reward(s, a, next_s, reward, done)
                    self.P[s][a][i] = (prob, next_s, new_reward, done)
    
    def train(self, max_iterations=1000, verbose=False):
        """
        Entrena l'agent mitjançant Iteració de Valor.
        
        Args:
            max_iterations: Nombre màxim d'iteracions
            verbose: Si s'ha de mostrar informació durant l'entrenament
            
        Returns:
            dict: Resultats de l'entrenament
        """
        # Registrem temps d'inici
        start_time = time.time()
        
        # Per registrar la història de delta (canvis en cada iteració)
        delta_history = []
        
        # Per registrar temps per iteració
        iteration_times = []
        
        if verbose:
            print(f"Iniciant Iteració de Valor amb gamma={self.gamma}, theta={self.theta}")
            print(f"Entorn: {self.nS} estats, {self.nA} accions")
            iterator = tqdm(range(max_iterations))
        else:
            iterator = range(max_iterations)
        
        # Iteració de Valor
        for i in iterator:
            iter_start_time = time.time()
            
            # Inicialitzem delta (canvi màxim en la funció de valor)
            delta = 0
            
            # Per cada estat
            for s in range(self.nS):
                # Guardem valor actual
                v = self.V[s]
                
                # Calculem valor de cada acció i prenem el màxim
                action_values = np.zeros(self.nA)
                
                for a in range(self.nA):
                    # Per cada possible resultat de l'acció
                    for prob, next_state, reward, done in self.P[s][a]:
                        # Actualitzem value d'acord amb l'equació de Bellman
                        if done:
                            action_values[a] += prob * reward
                        else:
                            action_values[a] += prob * (reward + self.gamma * self.V[next_state])
                
                # Actualitzem amb el màxim valor
                self.V[s] = np.max(action_values)
                
                # Actualitzem delta
                delta = max(delta, abs(v - self.V[s]))
            
            # Registrem temps i delta
            iter_time = time.time() - iter_start_time
            iteration_times.append(iter_time)
            delta_history.append(delta)
            
            # Comprovem convergència
            if delta < self.theta:
                if verbose:
                    print(f"Convergit després de {i+1} iteracions amb delta={delta}")
                break
        
        # Calculem la política i la funció Q òptimes
        for s in range(self.nS):
            # Calculem valors Q per a cada acció
            for a in range(self.nA):
                self.Q[s][a] = 0
                for prob, next_state, reward, done in self.P[s][a]:
                    if done:
                        self.Q[s][a] += prob * reward
                    else:
                        self.Q[s][a] += prob * (reward + self.gamma * self.V[next_state])
            
            # Escollim l'acció amb major valor Q
            self.policy[s] = np.argmax(self.Q[s])
        
        # Registrem temps total
        total_time = time.time() - start_time
        
        if verbose:
            print(f"Entrenament completat en {total_time:.4f} segons")
        
        return {
            "iterations": i + 1,
            "delta_history": delta_history,
            "iteration_times": iteration_times,
            "total_time": total_time,
            "convergence": delta < self.theta
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
            num_episodes: Nombre d'episodis a avaluar (augmentat a 500)
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
        plt.title(f'Funció de Valor (gamma={self.gamma})')
        
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
        plt.title(f'Política Òptima (gamma={self.gamma})')
        plt.tight_layout()
        
        # Guardar la imatge si s'ha especificat la ruta
        if save_path:
            # Crear el directori si no existeix
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"Política guardada a: {save_path}")
        
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
def run_experiments(output_dir="experiments_valor_iteracio"):
    # Crear directori per guardar resultats
    os.makedirs(output_dir, exist_ok=True)
    
    # Paràmetres a experimentar
    gamma_values = [0.9, 0.95, 0.99, 0.999]
    theta_values = [1e-3, 1e-4, 1e-6]
    reward_functions = {
        "default": default_reward,
        "step_penalty": step_penalty_reward,
        "cliff_avoidance": cliff_avoidance_reward
    }
    
    # Preparar DataFrame per resultats
    results = []
    
    # Generar totes les combinacions de paràmetres
    all_params = list(product(gamma_values, theta_values, reward_functions.items()))
    total_experiments = len(all_params)
    
    print(f"Iniciant {total_experiments} experiments...")
    
    # Executar experiments
    for i, (gamma, theta, (reward_name, reward_func)) in enumerate(all_params):
        print(f"Experiment {i+1}/{total_experiments}: gamma={gamma}, theta={theta}, reward={reward_name}")
        
        # Crear entorn
        env = gym.make('CliffWalking-v0', is_slippery=True)
        
        # Crear i entrenar agent
        agent = ValueIterationAgent(env, gamma=gamma, theta=theta, custom_reward=reward_func)
        train_results = agent.train(max_iterations=1000, verbose=False)
        
        # Avaluar agent amb 500 episodis en lloc de 20
        print(f"Avaluant agent amb 500 episodis...")
        eval_results = agent.evaluate(num_episodes=500)
        
        # Guardar resultats
        results.append({
            "gamma": gamma,
            "theta": theta,
            "reward_function": reward_name,
            "iterations": train_results["iterations"],
            "training_time": train_results["total_time"],
            "convergence": train_results["convergence"],
            "mean_reward": eval_results["mean_reward"],
            "std_reward": eval_results["std_reward"],
            "mean_steps": eval_results["mean_steps"],
            "std_steps": eval_results["std_steps"],
            "success_rate": eval_results["success_rate"]
        })
        
        # Si és l'experiment amb gamma=0.99, theta=1e-6 i recompensa per defecte,
        # guardem també la funció de valor i la política
        if gamma == 0.99 and theta == 1e-6 and reward_name == "default":
            save_path_value = os.path.join(output_dir, "valor_funcio_experiment.png")
            save_path_policy = os.path.join(output_dir, "politica_experiment.png")
            agent.visualize_value_function(save_path=save_path_value)
            agent.visualize_policy(save_path=save_path_policy)
        
        # Tancar entorn
        env.close()
    
    # Convertir a DataFrame
    results_df = pd.DataFrame(results)
    
    # Guardar a CSV
    csv_path = os.path.join(output_dir, "value_iteration_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Resultats guardats a '{csv_path}'")
    
    return results_df

# Funció per visualitzar resultats
def visualize_results(results_df=None, output_dir="experiments_valor_iteracio"):
    # Crear directori per guardar resultats
    os.makedirs(output_dir, exist_ok=True)
    
    if results_df is None:
        try:
            csv_path = os.path.join(output_dir, "value_iteration_results.csv")
            results_df = pd.read_csv(csv_path)
        except FileNotFoundError:
            print(f"No s'ha trobat el fitxer de resultats a {csv_path}. Executa primer els experiments.")
            return
    
    # Configurar estil
    sns.set(style="whitegrid")
    
    # Figura 1: Efecte de gamma en la recompensa mitjana per diferents funcions de recompensa
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=results_df, x="gamma", y="mean_reward", hue="reward_function", marker="o")
    plt.title("Efecte de gamma en la recompensa mitjana")
    plt.xlabel("Gamma")
    plt.ylabel("Recompensa mitjana")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "vi_gamma_vs_reward.png"))
    plt.show()
    
    # Figura 2: Temps d'entrenament vs iterations
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=results_df, x="iterations", y="training_time", hue="theta", size="gamma", sizes=(50, 200))
    plt.title("Temps d'entrenament vs Iteracions")
    plt.xlabel("Nombre d'iteracions")
    plt.ylabel("Temps d'entrenament (s)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "vi_iterations_vs_time.png"))
    plt.show()
    
    # Figura 3: Taxa d'èxit segons gamma i funció de recompensa
    plt.figure(figsize=(10, 6))
    sns.barplot(data=results_df, x="gamma", y="success_rate", hue="reward_function")
    plt.title("Taxa d'èxit segons gamma i funció de recompensa")
    plt.xlabel("Gamma")
    plt.ylabel("Taxa d'èxit")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "vi_gamma_vs_success.png"))
    plt.show()
    
    # Figura 4: Heatmap de recompensa mitjana vs gamma i theta
    pivot = results_df[results_df["reward_function"] == "default"].pivot_table(
        index="gamma", columns="theta", values="mean_reward"
    )
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, cmap="viridis", fmt=".2f")
    plt.title("Recompensa mitjana segons gamma i theta (recompensa per defecte)")
    plt.ylabel("Gamma")
    plt.xlabel("Theta")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "vi_heatmap_gamma_theta.png"))
    plt.show()
    
    # Figura 5: Distribució del nombre de passos segons la funció de recompensa
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=results_df, x="reward_function", y="mean_steps")
    plt.title("Distribució del nombre de passos segons la funció de recompensa")
    plt.xlabel("Funció de recompensa")
    plt.ylabel("Nombre mitjà de passos")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "vi_reward_vs_steps.png"))
    plt.show()

# Funció per executar l'algorisme
def run_value_iteration(gamma=0.99, theta=1e-6, render_evaluation=True, output_dir="experiments_valor_iteracio"):
    """
    Executa l'algorisme d'Iteració de Valor a l'entorn CliffWalking.
    
    Args:
        gamma: Factor de descompte
        theta: Llindar de convergència
        render_evaluation: Si s'ha de visualitzar l'avaluació
        output_dir: Directori on guardar els resultats
    """
    # Crear directori per guardar resultats
    os.makedirs(output_dir, exist_ok=True)
    
    # Crear l'entorn
    env = gym.make('CliffWalking-v0', is_slippery=True)
    
    # Crear i entrenar l'agent
    agent = ValueIterationAgent(env, gamma, theta)
    results = agent.train(verbose=True)
    
    # Visualitzar resultats
    print("Resultats de l'entrenament:")
    print(f"Iteracions: {results['iterations']}")
    print(f"Temps total: {results['total_time']:.4f} segons")
    
    # Visualitzar funció de valor i política
    save_path_value = os.path.join(output_dir, "valor_funcio.png")
    save_path_policy = os.path.join(output_dir, "politica.png")
    agent.visualize_value_function(save_path=save_path_value)
    agent.visualize_policy(save_path=save_path_policy)
    
    # Avaluar la política
    print("\nAvaluant la política...")
    if render_evaluation:
        render_env = gym.make('CliffWalking-v0', render_mode="human", is_slippery=True)
        agent_eval = ValueIterationAgent(render_env, gamma, theta)
        agent_eval.V = agent.V.copy()
        agent_eval.policy = agent.policy.copy()
        eval_results = agent_eval.evaluate(num_episodes=3, render=True)
        render_env.close()
    else:
        eval_results = agent.evaluate(num_episodes=500)
    
    print("Resultats de l'avaluació:")
    print(f"Recompensa mitjana: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"Passos mitjans: {eval_results['mean_steps']:.2f} ± {eval_results['std_steps']:.2f}")
    print(f"Taxa d'èxit: {eval_results['success_rate'] * 100:.2f}%")
    
    # Tancar l'entorn
    env.close()
    
    return agent, results, eval_results

# Funció principal
def main():
    # Crear directori per guardar resultats
    experiments_dir = "experiments_valor_iteracio"
    os.makedirs(experiments_dir, exist_ok=True)
    
    # Escollir mode
    print("1. Executar experiments complets")
    print("2. Visualitzar resultats existents")
    print("3. Executar un únic experiment amb gamma=0.99")
    mode = input("Escull una opció (1-3): ")
    
    if mode == "1":
        results_df = run_experiments(output_dir=experiments_dir)
        visualize_results(results_df, output_dir=experiments_dir)
    elif mode == "2":
        visualize_results(output_dir=experiments_dir)
    elif mode == "3":
        run_value_iteration(gamma=0.99, theta=1e-6, render_evaluation=False, output_dir=experiments_dir)
    else:
        print("Opció no vàlida")

if __name__ == "__main__":
    main()