import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def update_q(state, next_state, action, alpha, gamma):
    rsa = r[state, action]
    qsa = q[state, action]
    
    new_q = qsa + alpha * (rsa + gamma * max(q[next_state, :]) - qsa)
    q[state, action] = new_q
    # renormalize row to be between 0 and 1
    return r[state, action]

def show_path():
    # function to show the shortest path based on final Q 
    for i in range(len(q)):
        current_state = i
        path = "%i -> " % current_state
        n_steps = 0
        while current_state != 5 and n_steps < 20:
            next_state = np.argmax(q[current_state])
            current_state = next_state
            path += "%i -> " % current_state
            n_steps = n_steps + 1
        # cut off final arrow
        path = path[:-4]
        print("Path when starting from state %i" % i)
        print(path)
        print("")

# defines the reward/connection graph
r = np.array([[-1, -1, -1, -1,  0,  -1],
              [-1, -1, -1,  0, -1, 100],
              [-1, -1, -1,  0, -1,  -1],
              [-1,  0,  0, -1,  0,  -1],
              [ 0, -1, -1,  0, -1, 100],
              [-1,  0, -1, -1,  0, 100]]).astype("float32")
q = np.zeros_like(r)

gamma = 0.8
alpha = 1.
no_episodes = 50000
no_states = 6
no_actions = 6
epsilon = 0.9
rand_state = np.random.RandomState(1999)

for e in range(int(no_episodes)):
    states = list(range(no_states))
    rand_state.shuffle(states)
    current_state = states[0]
    goal = False
    if e % int(no_episodes / 1000) == 0 and e > 0: #will not pass if convergence: small deviation on q for 1000 episodes
        pass

    while not goal:
        # epsilon greedy exploration
        valid_moves = r[current_state] >= 0
        if rand_state.rand() < epsilon:
            actions = np.array(list(range(no_actions)))
            actions = actions[valid_moves == True]
            if type(actions) is int:
                actions = [actions]
            rand_state.shuffle(actions)
            action = actions[0]
            next_state = action
            if epsilon >= 0.5:
                epsilon = epsilon*0.99999
            else:
                epsilon = epsilon*0.9999
        else:
            if np.sum(q[current_state]) > 0:
                action = np.argmax(q[current_state])
                if epsilon >= 0.5:
                   epsilon = epsilon*0.99999
                else:
                   epsilon = epsilon*0.9999
            else:
                # Don't allow invalid moves at the start
                # Just take a random move
                actions = np.array(list(range(no_actions)))
                actions = actions[valid_moves == True]
                rand_state.shuffle(actions)
                action = actions[0]
            next_state = action
        reward = update_q(current_state, next_state, action,
                          alpha=alpha, gamma=gamma)
        # Goal state has reward 100
        if reward > 1:
            goal = True
        current_state = next_state
        
print(str(q))
show_path()