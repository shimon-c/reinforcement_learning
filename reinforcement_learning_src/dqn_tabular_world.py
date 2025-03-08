import numpy as np
import gym
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

# Custom GridWorld Environment
class GridWorld:
    def __init__(self, size=8):
        self.size = size
        self.state = (0, 0) # Start at top-left corner
        self.goal = (7, 7)
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)] # Right, Left, Down, Up
        self.action_space = len(self.actions)

    def reset(self):
        self.state = (0, 0)
        return self.state_to_index(self.state)

    def step(self, action):
        dx, dy = self.actions[action]
        x, y = self.state
        nx, ny = max(0, min(self.size - 1, x + dx)), max(0, min(self.size - 1, y + dy))
        self.state = (nx, ny)

        reward = 100 if self.state == self.goal else -1
        done = self.state == self.goal
        return self.state_to_index(self.state), reward, done

    def state_to_index(self, state):
        return state[0] * self.size + state[1]

env = GridWorld()

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

state_dim = 64
action_dim = env.action_space
policy_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
loss_fn = nn.MSELoss()


BUFFER_SIZE = 10000
BATCH_SIZE = 32
GAMMA = 0.99
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE = 10
MEMORY = deque(maxlen=BUFFER_SIZE)

def select_action(state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, action_dim - 1)
    with torch.no_grad():
        state_tensor = torch.tensor([state], dtype=torch.float32)
        return torch.argmax(policy_net(state_tensor)).item()

def train():
    if len(MEMORY) < BATCH_SIZE:
        return

    batch = random.sample(MEMORY, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)

    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
    next_q_values = target_net(next_states).max(1)[0]
    target_q_values = rewards + (1 - dones) * GAMMA * next_q_values

    loss = loss_fn(q_values, target_q_values.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Training loop
episodes = 1000
for episode in range(episodes):
    state = env.reset()
    epsilon = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
    done = False
    total_reward = 0

    while not done:
        action = select_action(state, epsilon)
        next_state, reward, done = env.step(action)
        MEMORY.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        train()

    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")

print("Training Complete!")



import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Extract Q-values for visualization
def get_q_values():
    q_values = np.zeros((8, 8))
    for x in range(8):
        for y in range(8):
            state_index = env.state_to_index((x, y))
            with torch.no_grad():
                q_values[x, y] = policy_net(torch.tensor([state_index], dtype=torch.float32)).max().item()
    return q_values

q_values = get_q_values()

# Plot Q-value heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(q_values, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Q-values Heatmap (Higher is Better)")
plt.xlabel("X-axis (Columns)")
plt.ylabel("Y-axis (Rows)")
plt.show()

def get_optimal_path():
    state = env.reset()
    path = [env.state]

    for _ in range(50): # Prevent infinite loops
        action = select_action(state, epsilon=0.0) # Greedy policy (no exploration)
        state, _, done = env.step(action)
        path.append(env.state)
        if done:
            break

    return path

path = get_optimal_path()

# Create Grid for visualization
grid = np.zeros((8, 8))
for (x, y) in path:
    grid[x, y] = 1 # Mark path

# Plot path
plt.figure(figsize=(8, 6))
sns.heatmap(grid, cmap="Greens", linewidths=0.5, cbar=False)
plt.title("Optimal Path from (0,0) to (7,7)")
plt.xlabel("X-axis (Columns)")
plt.ylabel("Y-axis (Rows)")
plt.show()

