import numpy as np

n_states = 16  # Number of states in the grid world
n_actions = 4  # Number of possible actions (up, down, left, right)
goal_state = 15  # Goal state

# Define the environment
class TabularModel:
    def __init__(self, n_states = 16,n_actions = 4,goal_state = 15):
        self.n_states = n_states
        self.n_actions = n_actions
        self.goal_state = goal_state
        self.Q_table = np.zeros((n_states, n_actions))

    def learn(self, learning_rate = 0.8, discount_factor = 0.95, exploration_prob = 0.2, epochs = 1000):
        # Q-learning algorithm
        Q_table = self.Q_table
        for epoch in range(epochs):
            # Select a random state
            current_state = np.random.randint(0, n_states)  # Start from a random state

            while current_state != goal_state:
                # Choose action with epsilon-greedy strategy
                if np.random.rand() < exploration_prob:
                    action = np.random.randint(0, n_actions)  # Explore
                else:
                    action = np.argmax(Q_table[current_state])  # Exploit

                # Simulate the environment (move to the next state)
                # For simplicity, move to the next state
                next_state = (current_state + 1) % n_states

                # Define a simple reward function (1 if the goal state is reached, 0 otherwise)
                reward = 1 if next_state == goal_state else 0

                # Update Q-value using the Q-learning update rule
                Q_table[current_state, action] += learning_rate * \
                    (reward + discount_factor *
                     np.max(Q_table[next_state]) - Q_table[current_state, action])

                current_state = next_state  # Move to the next state

        # After training, the Q-table represents the learned Q-values
        print("Learned Q-table:")
        print(Q_table)


if __name__ == '__main__':
    tab_model = TabularModel(n_states=n_states, n_actions=n_actions, goal_state=goal_state)
    learning_rate = 0.8
    discount_factor = 0.95
    exploration_prob = 0.2
    epochs = 1000
    tab_model.learn(learning_rate=learning_rate,
                    discount_factor=discount_factor,
                    exploration_prob=exploration_prob,epochs=epochs)
