import random
import numpy as np
import math

class MazePOMDP:
    def __init__(self, maze_size, observation_noise):
        self.maze_size = maze_size
        self.states = [(x, y) for x in range(maze_size) for y in range(maze_size)]
        self.actions = ["up", "down", "left", "right"]
        self.observations = [(x, y) for x in range(maze_size) for y in range(maze_size)]  # All possible positions
        self.observation_noise = observation_noise
        self.q_table = np.zeros((maze_size,maze_size))
        self.penalty = -10000000000   # in prectice we should get a
        self.actions = ["up", "down", "left", "right"]

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
        rd = 0
        if state == (self.maze_size - 1, self.maze_size - 1):  # Goal state
            rd = 10
        elif state in [(1, 1), (2, 2), (3, 3)]:  # Obstacles
            rd = -5
        else:
            rd = -1
        if action in ["left", "down"]:
            rd -= 1
        return rd


    def print_maze(self,agent_position):
        maze_size = self.maze_size
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

    def learn(self, belief=(0,0),num_simulations=1000):
        maze_size = self.maze_size
        for i in range(num_simulations):
            action = random.choice(self.actions)  # Random action selection
            next_state = self.transition(belief, action)
            observation = self.observation(belief, action, next_state)
            reward = self.reward(next_state, action)
            x,y = next_state
            self.q_table[x,y] += reward
            print("Step:", i + 1)
            print("Action:", action)
            print("Next State:", next_state)
            print("Observation:", observation)
            print("Reward:", reward)

            self.print_maze(next_state)
            print()

            belief = observation  # Update belief to the observed position
            # if next_state == (maze_size,maze_size):
            #     break

    def select_action(self,cur_state):
        x,y = cur_state
        rd = self.penalty
        best_act, next_state = None, None
        actions = ["right", "up"]
        max_coord = self.maze_size-1
        for act in actions:
            if act == "left" and y>0:
                cur_rd = self.q_table[x,y-1]
                if cur_rd>rd:
                    rd = cur_rd
                    best_act = act
                    next_state = (x,y-1)
            elif act == "right" and y<max_coord:
                cur_rd = self.q_table[x,y+1]
                if cur_rd>rd:
                    rd = cur_rd
                    best_act = act
                    next_state = (x,y+1)
            elif act == "down" and x>0:
                cur_rd = self.q_table[x-1,y]
                if cur_rd>rd:
                    rd = cur_rd
                    best_act = act
                    next_state = (x-1,y)
            elif act == "up" and x<max_coord:
                cur_rd = self.q_table[x + 1, y]
                if cur_rd > rd:
                    rd = cur_rd
                    best_act = act
                    next_state = (x + 1, y)
        return next_state, best_act, rd


    def print_solution(self,x=0,y=0):
        print(f'({x},{y}')
        rd = 0
        maze_size = self.maze_size
        cur_state = (x,y)
        k,max_k  = 0,maze_size*maze_size
        while x!=maze_size-1 or y!= maze_size-1:
            next_state, best_act, cur_rd = self.select_action(cur_state=cur_state)
            x,y = next_state
            rd += cur_rd
            self.print_maze(next_state)
            print('-----------------\n')
            cur_state = next_state
            k += 1
            if k > max_k:
                break
        print(f'reward:{rd}')



def main():
    maze_size = 5
    observation_noise = 0.2  # Noise level for observations
    pomdp = MazePOMDP(maze_size, observation_noise)
    num_simulations = 10
    belief = (0, 0)  # Initial belief (assume starting from (0, 0))

    pomdp.learn()
    pomdp.print_solution()
    # for i in range(num_simulations):
    #     action = random.choice(pomdp.actions)  # Random action selection
    #     next_state = pomdp.transition(belief, action)
    #     observation = pomdp.observation(belief, action, next_state)
    #     reward = pomdp.reward(next_state, action)
    #
    #     print("Step:", i + 1)
    #     print("Action:", action)
    #     print("Next State:", next_state)
    #     print("Observation:", observation)
    #     print("Reward:", reward)

        # print_maze(next_state, maze_size)
        # print()
        #
        # belief = observation  # Update belief to the observed position


if __name__ == "__main__":
    main()
