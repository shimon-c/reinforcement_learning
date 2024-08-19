import numpy as np
import math
import random

n_states = 16  # Number of states in the grid world
n_actions = 4  # Number of possible actions (up, down, left, right)
goal_state = 15  # Goal state

# Define the environment
class TabularModel:
    def __init__(self, n_states = 16,n_actions = 4,goal_state = 15):
        self.n_states = n_states
        self.n_actions = n_actions
        self.goal_state = goal_state
        self.tab_x_size = int(math.sqrt(n_states))
        self.Q_table = np.zeros((n_states, n_actions))
        self.Q_table[:,0] = -1
        self.act_dict = {0: 'left', 1: 'right', 2: 'up', 3: 'down'}
        self.inv_act_dict = {v:k for k,v in self.act_dict.items()}
        self.V_table = np.zeros((n_states,))

    def next_state(self, state, act):
        act_in = act
        act = self.act_dict[act_in]
        if act == 'left':
            if state%self.tab_x_size == 0:
                return state, act_in
            return state-1, act_in
        if act == 'up':
            if state>=self.tab_x_size:
                return state - self.tab_x_size,act_in
            else:
                return state, act_in
        if act == 'right': # and (state+1)%self.tab_x_size!=0:
            if (state+1)%self.tab_x_size!=0: # right side cannot move
                return state+1,act_in
            else:
                return state+1, act_in
        if act == 'down':
            if state + self.tab_x_size  <= self.goal_state:
                return state + self.tab_x_size,act_in
        return state,act_in

    def q_learn_simple(self, learning_rate = 0.8, discount_factor = 0.95, exploration_prob = 0.2, epochs = 1000):
        # Q-learning algorithm
        Q_table = self.Q_table
        down_index = self.inv_act_dict['down']
        right_index = self.inv_act_dict['right']
        for epoch in range(epochs):
            # Select a random state
            current_state = np.random.randint(0, n_states)  # Start from a random state

            while current_state != goal_state:
                # Choose action with epsilon-greedy strategy
                if np.random.rand() < exploration_prob:
                    action = np.random.randint(0, n_actions)  # Explore
                else:
                    action = np.argmax(Q_table[current_state])  # Exploit
                    if action <= 0:
                        action = random.choice([1,2,3])

                # Simulate the environment (move to the next state)
                # For simplicity, move to the next state
                # Need to be changed so next state depends on the transition
                next_state = (current_state + 1) % n_states
                #next_state = self.next_state(state=current_state, act=action)

                # Define a simple reward function (1 if the goal state is reached, 0 otherwise)
                reward = 1 if next_state == goal_state else 0
                #act = self.act_dict[action]
                #reward = self.reward_function(current_state,act,s_next=next_state)

                # Update Q-value using the Q-learning update rule
                Q_table[current_state, action] += learning_rate * \
                    (reward + discount_factor *
                     np.max(Q_table[next_state]) - Q_table[current_state, action])

                current_state = next_state  # Move to the next state

        # After training, the Q-table represents the learned Q-values
        print("Learned Q-table:")
        print(Q_table)
        self.print_q_table(Q_table=Q_table)


    def q_learn(self, learning_rate = 0.8, discount_factor = 0.95, exploration_prob = 0.2, epochs = 1000):
        # Q-learning algorithm
        Q_table = self.Q_table
        down_index = self.inv_act_dict['down']
        right_index = self.inv_act_dict['right']
        for epoch in range(epochs):
            # Select a random state
            current_state = np.random.randint(0, n_states)  # Start from a random state

            while current_state != goal_state:
                # Choose action with epsilon-greedy strategy
                if np.random.rand() < exploration_prob:
                    action = np.random.randint(1, n_actions)  # Explore
                else:
                    action = np.argmax(Q_table[current_state,])  # Exploit
                    if action <= 0:
                        action = down_index


                # Simulate the environment (move to the next state)
                # For simplicity, move to the next state
                # Need to be changed so next state depends on the transition
                #next_state = (current_state + 1) % n_states
                next_state, action = self.next_state(state=current_state, act=action)

                # Define a simple reward function (1 if the goal state is reached, 0 otherwise)
                #reward = 1 if next_state == goal_state else 0
                act = self.act_dict[action]
                reward = self.reward_function(current_state,act,s_next=next_state)

                # Update Q-value using the Q-learning update rule
                Q_table[current_state, action] += learning_rate * \
                    (reward + discount_factor *
                     np.max(Q_table[next_state]) - Q_table[current_state, action])

                current_state = next_state  # Move to the next state

        # After training, the Q-table represents the learned Q-values
        print("Learned Q-table:")
        print(Q_table)
        self.print_q_table(Q_table=Q_table)

    def print_q_table(self,Q_table):
        rows,cols = Q_table.shape
        q_str = 'Q_table'
        for r in range(rows):
            cur_row =''
            goal_state = r == self.goal_state
            for c in range(cols):
                a_max = np.argmax(Q_table[r,:])
                act = self.act_dict[a_max]
                if goal_state: act = 'nop'
                cur_row = f'{cur_row}\t {act}'
            q_str = f'{q_str}\n{cur_row}'
        print(q_str)

    def reward_function(self,s,a,s_next):
        if s == s_next:
            return 0
        if a == 'left' or a == 'up': return 0
        if s == self.goal_state:
            return 0
        if s_next==self.goal_state:
            if s+1 == self.goal_state and a == 'right':
                return 1
            if s+self.tab_x_size == self.goal_state and a=='down':
                return 1
        return 0

    def transition_model_uniform(self,s, a, s_next):
        if s+1 == s_next and a == 'right':
            return 0.25
        if s+self.tab_x_size == s_next and a == 'down':
            return 0.25
        if s-1 == s_next and a == 'left' and s%self.tab_x_size!=0:
            return 0.25
        return 0

    def transition_model(self,s, a, s_next):
        if s+1 == s_next and a == 'right':
            return 0.3
        if s+self.tab_x_size == s_next and a == 'down':
            return 0.6
        if s-1 == s_next and a == 'left' and s%self.tab_x_size!=0:
            return 0.1
        return 0

    def value_iteration(self, actions,  discount_factor = 0.95, epsilon=1e-16):
        # Initialize value function
        states = [k for k in range(self.n_states)]
        V = self.V_table
        V_prv = V.copy()
        epoch = 0
        while True:
            delta = 0
            for s in states:
                v = V[s]
                # V[s] = max(sum(self.transition_model(s, a, s_next) *
                #                (self.reward_function(s, a, s_next) + discount_factor * V_prv[s_next])
                #                for s_next in states) for a in actions)
                vals = []
                for a in actions:
                    val = 0
                    for s_next in states:
                        rv_s_a_s_next = self.reward_function(s, a, s_next)
                        v_next = discount_factor * V_prv[s_next]
                        cur_val = rv_s_a_s_next + v_next
                        cur_val_tr = cur_val*self.transition_model(s,a,s_next)
                        val += cur_val_tr
                    vals.append(val)
                max_val = max(vals)
                V[s] = max_val
                delta = max(delta, abs(v - V[s]))

            # Check for convergence
            if delta < epsilon:
                break
            V_prv = V.copy()
            epoch += 1
        # Extract optimal policy
        policy = {}
        for s in states:
            policy[s] = max(actions,
                            key=lambda a: sum(
                                self.transition_model(s, a, s_next) *
                                (self.reward_function(
                                    s, a, s_next) + discount_factor * V[s_next])
                                for s_next in states))
        print(f'Converged at epoch: {epoch}')
        V_ar = np.array(V)
        max_val = V_ar.max()
        V_ar[self.goal_state] += max_val
        # beautify can work only if n_stats is not a prime number
        V_ar = V_ar.reshape((-1, self.tab_x_size))
        nrows = self.n_states // self.tab_x_size
        ncols = self.tab_x_size
        policy_list = [[0 for k in range(ncols)] for kk in range(nrows)]
        for s,a in policy.items():
            r = s//ncols
            c = s%ncols
            policy_list[r][c] = a
        policy_list[-1][-1] = 'nop'
        return policy_list, V_ar

    def print_policy_list(self, policy):
        nr = len(policy)
        pol_str = 'Policy'
        for r in range(nr):
            pol_str = f'{pol_str}\n{policy[r]}'
        print(pol_str)


if __name__ == '__main__':
    tab_model = TabularModel(n_states=n_states, n_actions=n_actions, goal_state=goal_state)
    learning_rate = 0.8
    discount_factor = 0.95
    exploration_prob = 0.2
    epochs = 1000
    actions = [ 'left', 'right', 'up', 'down']
    policy, V = tab_model.value_iteration(actions=actions)
    print(f'V:\n{V}')
    tab_model.print_policy_list(policy=policy)
    tab_model.q_learn(learning_rate=learning_rate,
                      discount_factor=discount_factor,
                      exploration_prob=exploration_prob,epochs=epochs)
