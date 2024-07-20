import numpy as np
import gym
import math
from IPython import display

# Setup the environment
env = gym.make("CartPole-v1")

alpha = 0.1
gamma = 0.9
num_episodes = 150000

epsilon = 0.9
epsilon_decay_value = 0.99995

total_reward = 0
prior_reward = 0

Observation = [30, 30, 50, 50]
np_array_win_size = np.array([0.25, 0.25, 0.01, 0.1])

q_matrix = np.random.uniform(low=0, high=1, size=(Observation + [env.action_space.n]))


def get_discrete_state(state):
    discrete_state = state / np_array_win_size + np.array([15, 10, 1, 10])
    return tuple(discrete_state.astype(int))


def epsilon_greedy_action_selection(epsilon, q_values):
    if np.random.random() < epsilon:
        return np.random.randint(0, len(q_values))
    else:
        return np.argmax(q_values)


returns_sum = np.zeros(q_matrix.shape)
returns_count = np.zeros(q_matrix.shape)

# Monte Carlo implementation
for episode in range(num_episodes + 1):
    episode_transitions = []
    state = env.reset()
    done = False
    episode_reward = 0  # Store reward per episode

    while not done:
        state_index = get_discrete_state(state)
        action = epsilon_greedy_action_selection(0.1, q_matrix[state_index])
        next_state, reward, done, _ = env.step(action)
        episode_transitions.append((state_index, action, reward))
        episode_reward += reward
        state = next_state

    # Accumulate rewards and update average
    total_reward += episode_reward

    # Monte Carlo update at the end of the episode
    G = 0
    for transition in reversed(episode_transitions):
        state, action, reward = transition
        G = reward + gamma * G
        returns_sum[state + (action,)] += G
        returns_count[state + (action,)] += 1
        q_matrix[state + (action,)] = returns_sum[state + (action,)] / returns_count[state + (action,)]

    # Calculate and print average reward every 'print_every' episodes
    if episode % 100 == 0:
        mean_reward = total_reward / 100
        print(f"Episode: {episode}, Mean Reward: {mean_reward}, epsilon: {epsilon}")
        total_reward = 0
        if mean_reward > 195.0:
            print("Problem Solved!")
            break
        total_reward = 0  # Reset total rewards after printing


env.close()