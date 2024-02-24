import numpy as np
import gym
import random

# Definir la semilla para reproducibilidad
random.seed(42)
np.random.seed(42)

# Crear el entorno Frozen Lake
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)

# Mostrar la matriz de transición del entorno
print("Matriz de transición del entorno:")
print(env.env.P)

# Definir el agente
class Agent:
    def __init__(self, env):
        self.env = env
        self.V = np.zeros(env.observation_space.n)  # Valores de estado
        self.policy = np.zeros(env.observation_space.n)  # Política inicial: todas las acciones igualmente probables

    def value_iteration(self, gamma=0.9, epsilon=1e-10):
        # Iteración de valor
        while True:
            delta = 0
            for s in range(env.observation_space.n):
                v = self.V[s]
                self.V[s] = max([sum([p*(r + gamma*self.V[s_]) for (p, s_, r, _) in env.env.P[s][a]]) for a in range(env.action_space.n)])
                delta = max(delta, abs(v - self.V[s]))
            if delta < epsilon:
                break

    def extract_policy(self, gamma=0.9):
        # Extraer la política óptima de los valores de estado
        for s in range(env.observation_space.n):
            q_values = [sum([p*(r + gamma*self.V[s_]) for (p, s_, r, _) in env.env.P[s][a]]) for a in range(env.action_space.n)]
            self.policy[s] = np.argmax(q_values)

    def play_episodes(self, num_episodes=100):
        rewards = []
        for _ in range(num_episodes):
            state = env.reset()
            done = False
            total_reward = 0
            while not done:
                action = int(self.policy[state])
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                state = next_state
            rewards.append(total_reward)
        return rewards

# Crear un agente
agent = Agent(env)

# Ejecutar el proceso de iteración de valor
agent.value_iteration()

# Extraer la política óptima
agent.extract_policy()

# Jugar episodios usando la política óptima y mostrar las recompensas
rewards = agent.play_episodes()
print("Recompensas de los episodios jugados usando la política óptima:")
print(rewards)
