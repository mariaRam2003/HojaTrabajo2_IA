import numpy as np
class FrozenLake:
    def __init__(self, size=4, num_holes=3, seed=42):
        self.size = size
        self.num_holes = num_holes
        self.seed = seed
        np.random.seed(self.seed)
        self.grid = np.zeros((self.size, self.size), dtype=str)
        self.grid.fill('-')  # Inicialmente llenamos todo el tablero con '-'
        self.start = (0, 0)
        self.goal = (self.size - 1, self.size - 1)
        self.holes = self.generate_holes()
        self.agent_pos = self.start
        self.done = False

    def generate_holes(self):
        holes = set()
        while len(holes) < self.num_holes:
            x = np.random.randint(0, self.size)
            y = np.random.randint(0, self.size)
            if (x, y) != self.start and (x, y) != self.goal:
                holes.add((x, y))
        return holes

    def reset(self):
        self.agent_pos = self.start
        self.done = False
        self.update_grid()
        return self.agent_pos

    def update_grid(self):
        self.grid.fill('-')
        self.grid[self.goal] = 'G'
        for hole in self.holes:
            self.grid[hole] = 'H'
        self.grid[self.agent_pos] = '@'

    def step(self, action):
        if self.done:
            raise ValueError("Episode has terminated, please reset the environment.")

        x, y = self.agent_pos

        if action == 0:  # Up
            x = max(0, x - 1)
        elif action == 1:  # Down
            x = min(self.size - 1, x + 1)
        elif action == 2:  # Left
            y = max(0, y - 1)
        elif action == 3:  # Right
            y = min(self.size - 1, y + 1)
        else:
            raise ValueError("Invalid action.")

        if (x, y) in self.holes:
            reward = 0
            self.done = True
        elif (x, y) == self.goal:
            reward = 1
            self.done = True
        else:
            reward = 0

        self.agent_pos = (x, y)
        self.update_grid()
        return self.agent_pos, reward, self.done

# Example usage:
env = FrozenLake(size=4, num_holes=3, seed=42)

# Reiniciar el entorno para obtener una nueva episodio
agent_pos = env.reset()

print("Initial grid:")
print(env.grid)

print("Agent starts at:", env.start)
print("Goal is at:", env.goal)
print("Holes are at:", env.holes)

while not env.done:
    action = np.random.randint(4)
    new_pos, reward, done = env.step(action)
    print("Agent's new position:", new_pos)
    print("Reward received:", reward)
    print("Episode done?", done)
    print("Grid after action:")
    print(env.grid)

print("Episode finished!")
