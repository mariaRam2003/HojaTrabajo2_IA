import numpy as np

class FrozenLakeMDP:
    def __init__(self, size=4, num_holes=3, seed=42):
        self.size = size
        self.num_holes = num_holes
        self.seed = seed
        self.goal = (size - 1, size - 1)
        self.agent_pos = (0, 0)  # Agente comienza en la esquina superior izquierda
        self.generate_environment()

    def generate_environment(self):
        np.random.seed(self.seed)
        self.grid = np.zeros((self.size, self.size))
        self.grid[self.goal] = 1  # Posición de la meta
        self.holes = set()
        # Colocar agujeros aleatoriamente en todo el mapa
        while len(self.holes) < self.num_holes:
            x, y = np.random.randint(0, self.size, size=2)
            if (x, y) != self.goal and (x, y) != self.agent_pos:
                self.holes.add((x, y))
        for hole in self.holes:
            self.grid[hole] = -1  # Posición del agujero

    def print_initial_state(self):
        for i in range(self.size):
            for j in range(self.size):
                if (i, j) == self.goal:
                    print("G", end="\t")  # Posición de la meta
                elif (i, j) in self.holes:
                    print("H", end="\t")  # Posición del agujero
                elif (i, j) == self.agent_pos:
                    print("A", end="\t")  # Posición del agente
                else:
                    print(".", end="\t")  # Posición vacía
            print()  # Nueva línea después de cada fila

    def get_possible_actions(self, state):
        actions = []
        if state[0] > 0:
            actions.append('up')
        if state[0] < self.size - 1:
            actions.append('down')
        if state[1] > 0:
            actions.append('left')
        if state[1] < self.size - 1:
            actions.append('right')
        return actions

    def get_transition_probs(self, state, action):
        transition_probs = {}
        if state in self.holes:
            transition_probs[state] = {'prob': 1.0}
            return transition_probs
        possible_actions = self.get_possible_actions(state)
        if action not in possible_actions:
            raise ValueError("Invalid action")
        if action == 'up':
            next_state = (state[0] - 1, state[1])
        elif action == 'down':
            next_state = (state[0] + 1, state[1])
        elif action == 'left':
            next_state = (state[0], state[1] - 1)
        elif action == 'right':
            next_state = (state[0], state[1] + 1)
        if next_state == self.goal:
            transition_probs[next_state] = {'prob': 1.0}
        elif next_state in self.holes:
            transition_probs[next_state] = {'prob': 1.0}
        else:
            transition_probs[next_state] = {'prob': 0.8}
            remaining_prob = 0.2 / (len(possible_actions) - 1)
            for a in possible_actions:
                if a != action:
                    if a == 'up':
                        next_state = (state[0] - 1, state[1])
                    elif a == 'down':
                        next_state = (state[0] + 1, state[1])
                    elif a == 'left':
                        next_state = (state[0], state[1] - 1)
                    elif a == 'right':
                        next_state = (state[0], state[1] + 1)
                    if next_state == self.goal:
                        transition_probs[next_state] = {'prob': remaining_prob + 0.8}
                    elif next_state in self.holes:
                        transition_probs[next_state] = {'prob': remaining_prob + 0.8}
                    else:
                        transition_probs[next_state] = {'prob': remaining_prob}
        return transition_probs

    def policy_evaluation(self, policy, gamma=0.9, theta=1e-5):
        V = {s: 0 for s in np.ndindex(self.grid.shape)}
        while True:
            delta = 0
            for state in np.ndindex(self.grid.shape):
                if state == self.goal:
                    continue
                v = V[state]
                action = policy[state]
                transition_probs = self.get_transition_probs(state, action)
                V[state] = sum(transition_prob['prob'] * (reward + gamma * V[next_state])
                               for next_state, reward in self.get_expected_rewards(state, action).items()
                               for transition_prob in [transition_probs[next_state]])
                delta = max(delta, abs(v - V[state]))
            if delta < theta:
                break
        return V

    def policy_improvement(self, V, policy, gamma=0.9):
        policy_stable = True
        for state in np.ndindex(self.grid.shape):
            if state == self.goal:
                continue
            old_action = policy[state]
            action_values = {}
            for action in self.get_possible_actions(state):
                transition_probs = self.get_transition_probs(state, action)
                action_values[action] = sum(transition_prob['prob'] * (reward + gamma * V[next_state])
                                             for next_state, reward in self.get_expected_rewards(state, action).items()
                                             for transition_prob in [transition_probs[next_state]])
            policy[state] = max(action_values, key=action_values.get)
            if old_action != policy[state]:
                policy_stable = False
        return policy_stable

    def get_expected_rewards(self, state, action):
        transition_probs = self.get_transition_probs(state, action)
        return {next_state: self.grid[next_state] for next_state in transition_probs}

    def solve(self, gamma=0.9):
        policy = {(i, j): np.random.choice(self.get_possible_actions((i, j)))
                  for i in range(self.size) for j in range(self.size)}
        while True:
            V = self.policy_evaluation(policy, gamma)
            if self.policy_improvement(V, policy, gamma):
                break
        return policy, V

# Ejemplo de uso:
mdp = FrozenLakeMDP(size=4, num_holes=3, seed=42)
mdp.print_initial_state()
policy, values = mdp.solve()
print("Policy:")
for i in range(4):
    for j in range(4):
        state_number = i * 4 + j
        reward = values[(i, j)]
        terminal_state = values[(i, j)]
        action = policy[(i, j)]
        transition_probs = mdp.get_transition_probs((i, j), action)

        print(f"En el estado ({i}, {j}), la política óptima es tomar la acción '{action}'.")
        print(f"La recompensa esperada en este estado es aproximadamente {reward}.")
        print()
