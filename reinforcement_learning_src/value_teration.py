def value_iteration(states, actions, transition_model, reward_function, gamma, epsilon):
    # Initialize value function
    V = {s: 0 for s in states}

    while True:
        delta = 0
        for s in states:
            v = V[s]
            V[s] = max(sum(transition_model(s, a, s_next) *
                           (reward_function(s, a, s_next) + gamma * V[s_next])
                           for s_next in states) for a in actions)
            delta = max(delta, abs(v - V[s]))

        # Check for convergence
        if delta < epsilon:
            break

    # Extract optimal policy
    policy = {}
    for s in states:
        policy[s] = max(actions,
                        key=lambda a: sum(
                            transition_model(s, a, s_next) *
                            (reward_function(
                                s, a, s_next) + gamma * V[s_next])
                            for s_next in states))
    return policy, V
