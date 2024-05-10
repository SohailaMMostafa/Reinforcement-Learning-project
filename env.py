import gymnasium as gym


class Environment:
    def __init__(self, num_episodes):
        self.num_episodes = num_episodes
        self.env = gym.make('CartPole-v1', render_mode="human")

    def generate_episode(self):
        episode = {"states": [], "actions": [], "rewards": [], "total_reward": 0}  # Initialize total_reward
        for i in range(self.num_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                action = self.env.action_space.sample()  # Random action for demonstration
                next_state, reward, done, truncated, info = self.env.step(action)
                episode["states"].append(state)
                episode["actions"].append(action)
                episode["rewards"].append(reward)
                total_reward += reward  # Accumulate reward for this episode
                state = next_state
                if done:
                    break
            episode["total_reward"] = total_reward  # Assign total_reward to episode
            print(f"Episode {i + 1}, Total Reward: {total_reward}")
        return episode
