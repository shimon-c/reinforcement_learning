import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random

env_list = [[-1,-1,-50,-1,-1,-1,-1,-1],
        [-1,-50,-1,-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1,-1,-1,-1],
        [-1,-1,-50,-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1,-1,-1,100],
]

# Define the network architecture
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        state_size=2
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        self.tab_size = len(env_list)

    def norm(self, x):
        x_n = x / self.tab_size
        return x_n

    def forward(self, x):
        x = self.norm(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.squeeze(dim=1)
        return x

    def get_next_state(self, state):
        res = self(state)
        score = torch.max(res[0,:])
        act = torch.argmax(res[0,:])
        act = act.item()
        new_state = torch.zeros_like(state)
        new_state[:,:] = state
        if act == LEFT:
            new_state[0,1] -= 1
        elif act == RIGHT:
            new_state[0,1] += 1
        elif act == DOWN:
            new_state[0,0] -= 1
        else:
            new_state[0,0] += 1
        if new_state[0,0] >= self.tab_size:
            new_state[0,0] = self.tab_size-1
        if new_state[0,0]<0:
            new_state[0,0] = 0
        if new_state[0,1] >= self.tab_size:
            new_state[0, 1] = self.tab_size - 1
        if new_state[0,1]<0:
            new_state[0,1] = 0
        return new_state,score


    def get_path(self, x=0,y=0, device='cuda:0'):
        self.eval()
        state = torch.Tensor([y,x])
        state = state.reshape((1,-1))
        state = state.to(device)
        path = [(int(x),int(y))]
        done = False
        Y,X = env.get_shape()
        max_path = 3*Y*X
        score = 0
        while not done:
            next_state, cur_score = self.get_next_state(state)
            score += cur_score
            ny,nx = next_state[0,0].item(), next_state[0,1].item()
            path.append((int(nx),int(ny)))
            if nx==X-1 and ny==Y-1:
                done = True
            state = next_state
            if len(path)>max_path:
                done = True
        path_str = ''
        for pp in path:
            path_str = f'{path_str}->({pp[0]},{pp[1]})'
        path_str = f'{path_str}\nscore:{score}'
        print(path_str)
        return  path, path_str,score



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
        # In-place gradient clipping

        loss.backward()
        torch.nn.utils.clip_grad_value_(self.qnetwork_local.parameters(), 100)
        self.optimizer.step()

    def get_path(self):
        return self.qnetwork_local.get_path(0,0,device=self.device)

#import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


# Assume actions can be left,right, up, down So 4 actions.
LEFT=0
RIGHT=1
UP=2
DOWN=3
env_array = np.array(env_list)


class Enviroment:
    MAX_PENALTY = -1
    def __init__(self, arr=env_list, random_act=True):
        assert arr is not None
        self.env_arr = np.array(arr)
        self.action_array = np.array([0,1,2,3])
        self.random_act = random_act
        self.state = None

    def get_tensor(self, nx=None,ny=None):
        ten = torch.Tensor([ny,nx])
        ten = ten.reshape((1,-1))
        return ten

    def __call__(self, x=None,y=None, act:int=None) -> float:
        Y,X = self.env_arr.shape
        reward = -1
        # New state
        nx,ny=x,y
        yi,xi = int(y.item()),int(x.item())
        if act == LEFT:
            if x <= 0:
                return self.get_tensor(nx=nx,ny=ny),self.MAX_PENALTY,False
            reward = self.env_arr[yi,xi-1]
            ny,nx=yi,xi-1
        if act == RIGHT:
            if x >= X - 1:
                return self.get_tensor(nx=nx,ny=ny),self.MAX_PENALTY,False
            reward = self.env_arr[yi,xi+1]
            ny,nx=yi,xi+1
        if act == DOWN:
            if y <= 0:
                return self.get_tensor(nx=nx,ny=ny),self.MAX_PENALTY,False
            reward = self.env_arr[yi-1,xi]
            ny,nx=yi-1,xi
        if act == UP:
            if y >= Y - 1:
                return self.get_tensor(nx=nx,ny=ny),self.MAX_PENALTY,False
            reward = self.env_arr[yi+1,xi]
            ny,nx=yi+1,xi
        # episode is finished when we get to the terminal state
        terminate_stat = ny==Y-1 and nx==X-1
        new_state = self.get_tensor(nx=nx,ny=ny)
        return new_state, reward, terminate_stat

    def get_num_acts(self):
        return 4

    def get_state_size(self):
        Y,X = self.env_arr.shape
        return X*Y

    def sample_state(self):
        Y,X = self.env_arr.shape
        x = random.randint(0, X-1)
        y = random.randint(0, Y-1)
        return self.get_tensor(y,x)

    def sample_action_prv(self,state):
        n_actions = self.get_num_acts()
        reward = self.MAX_PENALTY-100
        best_act = -1
        y,x = int(state[0,0].item()), int(state[0,1].item())
        for act in range(0,n_actions):
            if act == LEFT and x>0:
                rwd = self.env_arr[y,x-1]
                if rwd > reward:
                    reward = rwd
                    best_act = LEFT
            elif act == RIGHT and x < self.env_arr.shape[1]:
                rwd = self.env_arr[y,x+1]
                if rwd > reward:
                    reward = rwd
                    best_act = RIGHT
            elif act == UP and y<self.env_arr.shape[0]:
                rwd = self.env_arr[y+1,x]
                if rwd > reward:
                    reward = rwd
                    best_act = UP
            elif y>0:
                rwd = self.env_arr[y - 1, x]
                if rwd > reward:
                    reward = rwd
                    best_act = DOWN
        #act = random.randint(0,n_actions-1)
        act = best_act
        return act

    def sample_action(self,state):
        n_actions = self.get_num_acts()
        np.random.shuffle(self.action_array)
        if self.random_act:
            return self.action_array[0]
        reward = self.MAX_PENALTY-100
        best_act = -1
        y,x = int(state[0,0].item()), int(state[0,1].item())
        for act in self.action_array:
            if act == LEFT and x>0:
                best_act = LEFT
            elif act == RIGHT and x < self.env_arr.shape[1]:
                best_act = RIGHT
            elif act == UP and y<self.env_arr.shape[0]:
                best_act = UP
            elif y>0:
                best_act = DOWN
            if act>=0:
                break
        #act = random.randint(0,n_actions-1)
        act = best_act
        return act

    def reset(self):
        self.state =  self.get_tensor(ny=0,nx=0)
        return self.state

    #observation, reward, terminated, truncated, _ = env.step(action.item())
    def step(self, action=None):
        state = self.state
        x,y = state[0,1], state[0,0]
        nstate, reward, terminate_stat = self(x=x, y=y,act=action)
        if (nstate[0,0]<0 or nstate[0,0] >= self.env_arr.shape[0] or
                nstate[0,1]<0 or nstate[0,1] >= self.env_arr.shape[1]):
            print("Bug")
        # remeber new state
        self.state = nstate
        return nstate, reward, terminate_stat,0,0

    def get_shape(self):
        return self.env_arr.shape

    def eval_path(self, path=[]):
        score = 0
        for p in path:
            pass


#from dqn import DQNAgent

# Create the environment
#env = gym.make('CartPole-v1')
env = Enviroment()
# Get the state and action sizes
state_size = 2      # env.observation_space.shape[0]
action_size = 4     #env.action_space.n

# Set the random seed
seed = 0

# Create the DQN agent
agent = DQNAgent(state_size, action_size, seed)

# Set the number of episodes and the maximum number of steps per episode
num_episodes = 3000
max_steps = 2000

# Set the exploration rate
eps = eps_start = 1.0
eps_end = 0.01
eps_decay = 0.995
def compute_decay(e_start, e_end, num_episode):
    decay = math.exp(math.log(eps_end/e_start)/(num_episode-1))
    return decay

decay = compute_decay(e_start=eps, e_end=eps_end,num_episode=num_episodes)

# Set the rewards and scores lists
rewards = []
scores = []


# Training
# Run the training loop
for i_episode in range(num_episodes):
    print(f'Episode: {i_episode}')
    # Initialize the environment and the state
    state = env.reset()
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
import seaborn as sns
def show_heat_map(path):
    NY, NX = env.get_shape()
    grid = np.zeros((NY, NX))
    for pt in path:
        x,y = pt
        grid[x, y] = 1  # Mark path

    # Plot path
    plt.figure(figsize=(8, 6))
    sns.heatmap(grid, cmap="Greens", linewidths=0.5, cbar=False)
    plt.title(f"Optimal Path from (0,0) to ({NY},{NX})")
    plt.xlabel("X-axis (Columns)")
    plt.ylabel("Y-axis (Rows)")



path = agent.get_path()
show_heat_map(path)
plt.ylabel("Score")
plt.xlabel("Episode")
plt.plot(range(len(rewards)), rewards)
plt.plot(range(len(rewards)), scores)
plt.legend(['Reward', "Score"])
plt.show()