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

# SARSA Learning
for episode in range(num_episodes + 1):
    state_index = get_discrete_state(env.reset())
    done = False
    episode_reward = 0

    # Choose action using epsilon-greedy strategy
    if np.random.random() > epsilon:
        action = np.argmax(q_matrix[state_index])
    else:
        action = np.random.randint(0, env.action_space.n)

    while not done:
        new_state, reward, done, _ = env.step(action)
        episode_reward += reward

        new_state_index = get_discrete_state(new_state)

        # Choose the next action (again using epsilon-greedy)
        if np.random.random() > epsilon:
            new_action = np.argmax(q_matrix[new_state_index])
        else:
            new_action = np.random.randint(0, env.action_space.n)

        # SARSA update
        current_q = q_matrix[state_index + (action,)]
        next_q = q_matrix[new_state_index + (new_action,)]
        error = reward + gamma * (next_q - current_q)
        new_q = q_matrix[state_index + (action,)] + alpha * error
        q_matrix[state_index + (action,)] = new_q

        state_index = new_state_index
        action = new_action

    # Epsilon decay
    if epsilon > 0.05:
        if episode_reward > prior_reward and episode > 100:
            epsilon = math.pow(epsilon_decay_value, episode - 100)

    total_reward += episode_reward
    prior_reward = episode_reward

    if episode % 100 == 0:
        mean_reward = total_reward / 100
        print(f"Episode: {episode}, Mean Reward: {mean_reward}, epsilon: {epsilon}")
        total_reward = 0
        if mean_reward > 195.0:
            print("Problem Solved!")
            break


env.close()