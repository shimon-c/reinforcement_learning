import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random


# Define the network architecture
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Define the replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.index = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.index] = (state, action, reward, next_state, done)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in batch:
            state, action, reward, next_state, done = self.buffer[i]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        return (
            torch.tensor(np.array(states)).float(),
            torch.tensor(np.array(actions)).long(),
            torch.tensor(np.array(rewards)).unsqueeze(1).float(),
            torch.tensor(np.array(next_states)).float(),
            torch.tensor(np.array(dones)).unsqueeze(1).int()
        )

    def __len__(self):
        return len(self.buffer)


# Define the Vanilla DQN agent
class DQNAgent:
    def __init__(self, state_size, action_size, seed, learning_rate=1e-3, capacity=1000000,
                 discount_factor=0.99, update_every=4, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.update_every = update_every
        self.batch_size = batch_size
        self.steps = 0

        self.qnetwork_local = QNetwork(state_size, action_size)
        self.device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.qnetwork_local = self.qnetwork_local.to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(capacity)

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay buffer
        self.replay_buffer.push(state, action, reward, next_state, done)

        # Learn every update_every steps
        self.steps += 1
        if self.steps % self.update_every == 0:
            if len(self.replay_buffer) > self.batch_size:
                experiences = self.replay_buffer.sample(self.batch_size)
                self.learn(experiences)

    def act(self, state, eps=0.0):
        # Epsilon-greedy action selection
        if random.random() > eps:
            state = torch.tensor(state).float().unsqueeze(0).to(self.device)
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local(state)
            self.qnetwork_local.train()
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from local model
        next_states = next_states.to(self.device)
        Q_targets_next = self.qnetwork_local(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        rewards =  rewards.to(self.device)
        dones = dones.to(self.device)
        Q_targets = rewards + (self.discount_factor * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        states = states.to(self.device)
        qnet_out = self.qnetwork_local(states)
        actions = actions.to(self.device)
        Q_expected = qnet_out.gather(1, actions.view(-1, 1))

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()


import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

#from dqn import DQNAgent

# Create the environment
env = gym.make('CartPole-v1')

# Get the state and action sizes
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Set the random seed
seed = 0

# Create the DQN agent
agent = DQNAgent(state_size, action_size, seed)

# Set the number of episodes and the maximum number of steps per episode
num_episodes = 1000
max_steps = 1000

# Set the exploration rate
eps = eps_start = 1.0
eps_end = 0.01
eps_decay = 0.995

# Set the rewards and scores lists
rewards = []
scores = []


# Training
# Run the training loop
for i_episode in range(num_episodes):
    print(f'Episode: {i_episode}')
    # Initialize the environment and the state
    state = env.reset()[0]
    score = 0
    # eps = eps_end + (eps_start - eps_end) * np.exp(-i_episode / eps_decay)
    # Update the exploration rate
    eps = max(eps_end, eps_decay * eps)

    # Run the episode
    for t in range(max_steps):
        # Select an action and take a step in the environment
        action = agent.act(state, eps)
        next_state, reward, done, trunc, _ = env.step(action)
        # Store the experience in the replay buffer and learn from it
        agent.step(state, action, reward, next_state, done)
        # Update the state and the score
        state = next_state
        score += reward
        # Break the loop if the episode is done
        if done or trunc:
            break

    print(f"\tScore: {score}, Epsilon: {eps}")
    # Save the rewards and scores
    rewards.append(score)
    scores.append(np.mean(rewards[-100:]))

# Close the environment
env.close()

plt.ylabel("Score")
plt.xlabel("Episode")
plt.plot(range(len(rewards)), rewards)
plt.plot(range(len(rewards)), scores)
plt.legend(['Reward', "Score"])
plt.show()