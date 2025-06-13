import random

from game_model import GameModel
import abc
import functools
import itertools

import numpy as np


class SolutionConcept(abc.ABC):
    @abc.abstractmethod
    def solution_policy(self, agent_id, state, game, q_table):
        pass


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class ParetoSolutionConcept(SolutionConcept):
    def is_dominated(self, joint_action, other_joint_action, state, game, q_table):
        strictly_better_for_at_least_one = False
        for agent_id in range(game.num_agents):
            joint_action_index = game.action_space_index[joint_action]
            other_joint_action_index = game.action_space_index[other_joint_action]
            if q_table[agent_id][state][other_joint_action_index] < q_table[agent_id][state][joint_action_index]:
                return False
            if q_table[agent_id][state][other_joint_action_index] > q_table[agent_id][state][joint_action_index]:
                strictly_better_for_at_least_one = True
        return strictly_better_for_at_least_one

    def find_pareto_efficient_solutions(self, state, game, q_table):
        joint_actions = list(itertools.product(range(game.num_actions), repeat=game.num_agents))
        pareto_solutions = []

        for joint_action in joint_actions:
            dominated = False
            for other_joint_action in joint_actions:
                if other_joint_action == joint_action:
                    continue
                if self.is_dominated(joint_action, other_joint_action, state, game, q_table):
                    dominated = True
                    break
            if not dominated:
                pareto_solutions.append(joint_action)

        return pareto_solutions

    def solution_policy(self, agent_id, state, game, q_table):
        pareto_solutions = self.find_pareto_efficient_solutions(state, game, q_table)
        if len(pareto_solutions) > 0:
            action_counts = np.zeros(game.num_actions)
            for pareto_solution in pareto_solutions:
                action_counts[pareto_solution[agent_id]] += 1
            probs = action_counts / len(pareto_solutions)
            return probs
        else:
            uniform_distribution = np.ones(game.num_actions)
            return uniform_distribution / np.sum(uniform_distribution)


class MinimaxSolutionConcept(SolutionConcept):
    def opponent_max_values(self, agent_id, state, game, q_table):
        action_scores = []
        for action in range(game.num_actions):
            max_opponent_payoff = float('-inf')
            for opponent_action in range(game.num_actions):
                if agent_id == 0:  # Suponemos sólo dos agentes
                    joint_action = (action, opponent_action)
                else:
                    joint_action = (opponent_action, action)
                joint_action_index = game.action_space_index[joint_action]
                score = q_table[1 - agent_id][state][joint_action_index]
                if score > max_opponent_payoff:
                    max_opponent_payoff = score
            action_scores.append(max_opponent_payoff)
        return np.array(action_scores)

    def solution_policy(self, agent_id, state, game, q_table):
        vals = np.array(self.opponent_max_values(agent_id, state, game, q_table))
        return softmax(-vals)


class NashSolutionConcept(SolutionConcept):
    def find_nash_equilibria(self, state, game, q_table):
        best_responses = [self.calculate_best_responses(i, state, game, q_table) for i in range(game.num_agents)]
        return list(set.intersection(*best_responses))

    @functools.cache
    def generate_others_actions(self, fixed_agent_id, num_agents, num_actions):
        # Lista con las estrategias que pueden seguir los demás agentes
        strategies_minus_me = [range(num_actions) if i != fixed_agent_id else [None]
                               for i in range(num_agents)]
        # itertools.product nos da el producto cartesiano
        return list([list(t) for t in itertools.product(*strategies_minus_me)])

    def calculate_best_responses(self, agent_id, state, game, q_table):
        others_joint_actions = self.generate_others_actions(agent_id, game.num_agents, game.num_actions)
        best_responses = []
        for joint_action in others_joint_actions:
            max_payoff = float('-inf')
            best_response = None
            for action in range(game.num_actions):
                joint_action[agent_id] = action
                full_joint_action = tuple(joint_action)
                joint_action_index = game.action_space_index[full_joint_action]
                payoff = q_table[agent_id][state][joint_action_index]
                if payoff > max_payoff:
                    max_payoff = payoff
                    best_response = full_joint_action
            best_responses.append(best_response)
        return set(best_responses)

    def solution_policy(self, agent_id, state, game, q_table):
        nash_equilibria = self.find_nash_equilibria(state, game, q_table)
        if len(nash_equilibria) > 0:
            action_counts = np.zeros(game.num_actions)
            for equilibrium in nash_equilibria:
                action_counts[equilibrium[agent_id]] += 1
            probs = action_counts / len(nash_equilibria)
            return probs
        else:
            uniform_distribution = np.ones(game.num_actions)
            return uniform_distribution / np.sum(uniform_distribution)


class WelfareSolutionConcept(SolutionConcept):
    def find_welfare_maximizing_solutions(self, state, game, q_table):
        joint_actions = list(itertools.product(range(game.num_actions), repeat=game.num_agents))
        welfare_values = []

        for joint_action in joint_actions:
            welfare = self.calculate_welfare(joint_action, state, game, q_table)
            welfare_values.append((welfare, joint_action))

        max_welfare = max(welfare_values, key=lambda x: x[0])[0]
        welfare_maximizing_solutions = [action for welfare, action in welfare_values if welfare == max_welfare]

        return welfare_maximizing_solutions

    def calculate_welfare(self, joint_action, state, game, q_table):
        welfare = 0
        joint_action_index = game.action_space_index[joint_action]
        for agent_id in range(game.num_agents):
            welfare += q_table[agent_id][state][joint_action_index]
        return welfare

    def solution_policy(self, agent_id, state, game, q_table):
        welfare_solutions = self.find_welfare_maximizing_solutions(state, game, q_table)
        num_solutions = len(welfare_solutions)

        if num_solutions > 0:
            # Initialize probability distribution for the agent's actions
            probs = np.zeros(game.num_actions)

            # Calculate the probability assigned to each action
            for solution in welfare_solutions:
                probs[solution[agent_id]] += 1 / num_solutions

            return probs
        else:
            uniform_distribution = np.ones(game.num_actions)
            return uniform_distribution / np.sum(uniform_distribution)


