import numpy as np

class FrozenLakeMDP:
    def __init__(self, size=4, num_holes=3, seed=42):
        """
        Inicializa un entorno Frozen Lake MDP con parámetros específicos.

        Args:
            size (int): Tamaño del entorno (por defecto es 4x4).
            num_holes (int): Número de agujeros en el entorno.
            seed (int): Semilla para la generación de números aleatorios.
        """
        self.size = size
        self.num_holes = num_holes
        self.seed = seed
        self.goal = (size - 1, size - 1)  # Posición de la meta en la esquina inferior derecha
        self.agent_pos = (0, 0)  # Posición inicial del agente en la esquina superior izquierda
        self.generate_environment()  # Genera el entorno Frozen Lake

    def generate_environment(self):
        """
        Genera el entorno Frozen Lake, colocando aleatoriamente la meta, los agujeros y el agente.
        """
        np.random.seed(self.seed)
        self.grid = np.zeros((self.size, self.size))  # Crea una cuadrícula de tamaño específico
        self.grid[self.goal] = 1  # Coloca la meta en la posición de la meta
        self.holes = set()  # Conjunto para almacenar las posiciones de los agujeros
        # Coloca los agujeros aleatoriamente en la cuadrícula
        while len(self.holes) < self.num_holes:
            x, y = np.random.randint(0, self.size, size=2)
            if (x, y) != self.goal and (x, y) != self.agent_pos:
                self.holes.add((x, y))
        # Marca las posiciones de los agujeros en la cuadrícula
        for hole in self.holes:
            self.grid[hole] = -1

    def print_initial_state(self):
        """
        Imprime el estado inicial del entorno Frozen Lake.
        """
        for i in range(self.size):
            for j in range(self.size):
                if (i, j) == self.goal:
                    print("G", end="\t")  # Posición de la meta
                elif (i, j) in self.holes:
                    print("H", end="\t")  # Posición de los agujeros
                elif (i, j) == self.agent_pos:
                    print("A", end="\t")  # Posición del agente
                else:
                    print(".", end="\t")  # Posición vacía
            print()

    def get_possible_actions(self, state):
        """
        Obtiene las acciones posibles que el agente puede tomar en un estado dado.

        Args:
            state (tuple): Coordenadas del estado en la cuadrícula.

        Returns:
            list: Lista de acciones posibles ('up', 'down', 'left', 'right').
        """
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
        """
        Obtiene las probabilidades de transición para el siguiente estado dado un estado y una acción.

        Args:
            state (tuple): Coordenadas del estado actual en la cuadrícula.
            action (str): Acción tomada por el agente ('up', 'down', 'left', 'right').

        Returns:
            dict: Diccionario de probabilidades de transición para el siguiente estado.
        """
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
        if next_state == self.goal or next_state in self.holes:
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
                    if next_state == self.goal or next_state in self.holes:
                        transition_probs[next_state] = {'prob': remaining_prob + 0.8}
                    else:
                        transition_probs[next_state] = {'prob': remaining_prob}
        return transition_probs

    def policy_evaluation(self, policy, gamma=0.9, theta=1e-5):
        """
        Evalúa una política dada utilizando el algoritmo de evaluación de política iterativa.

        Args:
            policy (dict): Política actual (estado -> acción).
            gamma (float): Factor de descuento para futuras recompensas (por defecto es 0.9).
            theta (float): Umbral de convergencia (por defecto es 1e-5).

        Returns:
            dict: Valores de utilidad para cada estado.
        """
        V = {s: 0 for s in np.ndindex(self.grid.shape)}  # Inicializa los valores de utilidad
        while True:
            delta = 0
            for state in np.ndindex(self.grid.shape):
                if state == self.goal:
                    continue
                v = V[state]
                action = policy[state]
                transition_probs = self.get_transition_probs(state, action)
                # Calcula el valor de utilidad para el estado actual
                V[state] = sum(transition_prob['prob'] * (reward + gamma * V[next_state])
                               for next_state, reward in self.get_expected_rewards(state, action).items()
                               for transition_prob in [transition_probs[next_state]])
                delta = max(delta, abs(v - V[state]))  # Calcula el cambio máximo en los valores
            if delta < theta:  # Comprueba la convergencia
                break
        return V

    def policy_improvement(self, V, policy, gamma=0.9):
        """
        Mejora una política dada utilizando los valores de utilidad calculados.

        Args:
            V (dict): Valores de utilidad para cada estado.
            policy (dict): Política actual (estado -> acción).
            gamma (float): Factor de descuento para futuras recompensas (por defecto es 0.9).

        Returns:
            bool: True si la política es estable, False de lo contrario.
        """
        policy_stable = True
        for state in np.ndindex(self.grid.shape):
            if state == self.goal:
                continue
            old_action = policy[state]
            action_values = {}
            for action in self.get_possible_actions(state):
                transition_probs = self.get_transition_probs(state, action)
                # Calcula el valor de utilidad para cada acción posible
                action_values[action] = sum(transition_prob['prob'] * (reward + gamma * V[next_state])
                                            for next_state, reward in
                                            self.get_expected_rewards(state, action).items()
                                            for transition_prob in [transition_probs[next_state]])
            # Selecciona la acción con el mayor valor de utilidad
            policy[state] = max(action_values, key=action_values.get)
            if old_action != policy[state]:
                policy_stable = False  # La política no es estable si ha cambiado
        return policy_stable

    def get_expected_rewards(self, state, action):
        """
        Obtiene las recompensas esperadas para el próximo estado dado un estado y una acción.

        Args:
            state (tuple): Coordenadas del estado actual en la cuadrícula.
            action (str): Acción tomada por el agente ('up', 'down', 'left', 'right').

        Returns:
            dict: Diccionario de recompensas esperadas para el próximo estado.
        """
        transition_probs = self.get_transition_probs(state, action)
        return {next_state: self.grid[next_state] for next_state in transition_probs}

    def solve(self, gamma=0.9):
        """
        Resuelve el MDP Frozen Lake para encontrar la política óptima y los valores de utilidad.

        Args:
            gamma (float): Factor de descuento para futuras recompensas (por defecto es 0.9).

        Returns:
            tuple: Política óptima (estado -> acción) y valores de utilidad (estado -> valor).
        """
        # Política inicial: selección aleatoria de acciones para cada estado
        policy = {(i, j): np.random.choice(self.get_possible_actions((i, j)))
                  for i in range(self.size) for j in range(self.size)}
        while True:
            V = self.policy_evaluation(policy, gamma)  # Evalúa la política actual
            if self.policy_improvement(V, policy, gamma):  # Mejora la política
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

