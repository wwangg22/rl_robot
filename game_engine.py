import numpy as np

class Game2048:
    def __init__(self):
        self.grid_size = 4
        self.reset()

    def reset(self):
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=float)  # Use float
        self.add_new_tile()
        self.add_new_tile()
        return self.grid.flatten()

    def add_new_tile(self):
        empty_cells = list(zip(*np.where(self.grid == 0)))
        if empty_cells:
            row, col = empty_cells[np.random.randint(len(empty_cells))]
            self.grid[row, col] = 2.0 if np.random.rand() < 0.9 else 4.0  # Float values

    def compress(self, grid):
        new_grid = np.zeros_like(grid)
        for i in range(self.grid_size):
            pos = 0
            for j in range(self.grid_size):
                if grid[i, j] != 0:
                    new_grid[i, pos] = grid[i, j]
                    pos += 1
        return new_grid

    def merge(self, grid):
        reward = 0
        for i in range(self.grid_size):
            for j in range(self.grid_size - 1):
                if grid[i, j] == grid[i, j + 1] and grid[i, j] != 0:
                    grid[i, j] *= 2.0  # Float values
                    reward += grid[i, j]
                    grid[i, j + 1] = 0
        return grid, reward

    def reverse(self, grid):
        return np.flip(grid, axis=1)

    def transpose(self, grid):
        return np.transpose(grid)

    def move(self, direction):
        if direction == 0:  # Up
            self.grid = self.transpose(self.grid)
            self.grid = self.compress(self.grid)
            self.grid, reward = self.merge(self.grid)
            self.grid = self.compress(self.grid)
            self.grid = self.transpose(self.grid)
        elif direction == 1:  # Down
            self.grid = self.transpose(self.grid)
            self.grid = self.reverse(self.grid)
            self.grid = self.compress(self.grid)
            self.grid, reward = self.merge(self.grid)
            self.grid = self.compress(self.grid)
            self.grid = self.reverse(self.grid)
            self.grid = self.transpose(self.grid)
        elif direction == 2:  # Left
            self.grid = self.compress(self.grid)
            self.grid, reward = self.merge(self.grid)
            self.grid = self.compress(self.grid)
        elif direction == 3:  # Right
            self.grid = self.reverse(self.grid)
            self.grid = self.compress(self.grid)
            self.grid, reward = self.merge(self.grid)
            self.grid = self.compress(self.grid)
            self.grid = self.reverse(self.grid)
        else:
            raise ValueError("Invalid direction! Use 0 (up), 1 (down), 2 (left), or 3 (right).")

        return reward

    def is_game_over(self):
        if np.any(self.grid == 0):  # Empty cells
            return False
        for i in range(self.grid_size):
            for j in range(self.grid_size - 1):
                if self.grid[i, j] == self.grid[i, j + 1]:  # Adjacent horizontal match
                    return False
        for i in range(self.grid_size - 1):
            for j in range(self.grid_size):
                if self.grid[i, j] == self.grid[i + 1, j]:  # Adjacent vertical match
                    return False
        # print("max value: ", np.max(self.grid))
        return True

    def step(self, action):
        initial_grid = self.grid.copy()
        reward = self.move(action)
        if not np.array_equal(initial_grid, self.grid):
            self.add_new_tile()

        done = self.is_game_over()
        if done:
            reward += np.max(self.grid)
        return self.grid.flatten(), reward, done

    def get_observation_space(self):
        # Returns the shape of the observation space (flattened grid)
        return (self.grid_size * self.grid_size,)

    def get_input_space(self):
        # Returns the discrete action space (4 actions: up, down, left, right)
        return 4

# Example usage:
if __name__ == "__main__":
    game = Game2048()
    print("Observation Space:", game.get_observation_space())
    print("Input Space:", game.get_input_space())
    print("Initial State:")
    print(game.grid)
    done = False
    while not done:
        action = int(input("Enter action (0=Up, 1=Down, 2=Left, 3=Right): "))
        state, reward, done = game.step(action)
        print(f"Reward: {reward}")
        print(state.reshape(4, 4))
    print("Game Over!")
