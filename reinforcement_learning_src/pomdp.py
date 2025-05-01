import random


class MazePOMDP:
    def __init__(self, maze_size, observation_noise):
        self.maze_size = maze_size
        self.states = [(x, y) for x in range(maze_size) for y in range(maze_size)]
        self.actions = ["up", "down", "left", "right"]
        self.observations = [(x, y) for x in range(maze_size) for y in range(maze_size)]  # All possible positions
        self.observation_noise = observation_noise

    def transition(self, state, action):
        x, y = state
        if action == "up":
            return (max(x - 1, 0), y)
        elif action == "down":
            return (min(x + 1, self.maze_size - 1), y)
        elif action == "left":
            return (x, max(y - 1, 0))
        elif action == "right":
            return (x, min(y + 1, self.maze_size - 1))

    def observation(self, state, action, next_state):
        if random.random() < self.observation_noise:
            return next_state  # Noisy observation is the true position
        else:
            return random.choice(self.observations)  # Random position as noisy observation

    def reward(self, state, action):
        if state == (self.maze_size - 1, self.maze_size - 1):  # Goal state
            return 10
        elif state in [(1, 1), (2, 2), (3, 3)]:  # Obstacles
            return -5
        else:
            return -1


def print_maze(agent_position, maze_size):
    for i in range(maze_size):
        for j in range(maze_size):
            if (i, j) == agent_position:
                print("A", end=" ")  # Agent
            elif (i, j) == (maze_size - 1, maze_size - 1):
                print("G", end=" ")  # Goal
            elif (i, j) in [(1, 1), (2, 2), (3, 3)]:
                print("X", end=" ")  # Obstacle
            else:
                print(".", end=" ")  # Empty space
        print()


def main():
    maze_size = 5
    observation_noise = 0.2  # Noise level for observations
    pomdp = MazePOMDP(maze_size, observation_noise)
    num_simulations = 10
    belief = (0, 0)  # Initial belief (assume starting from (0, 0))

    for i in range(num_simulations):
        action = random.choice(pomdp.actions)  # Random action selection
        next_state = pomdp.transition(belief, action)
        observation = pomdp.observation(belief, action, next_state)
        reward = pomdp.reward(next_state, action)

        print("Step:", i + 1)
        print("Action:", action)
        print("Next State:", next_state)
        print("Observation:", observation)
        print("Reward:", reward)

        print_maze(next_state, maze_size)
        print()

        belief = observation  # Update belief to the observed position


if __name__ == "__main__":
    main()
