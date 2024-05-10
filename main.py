from env import Environment

num_episodes = 5  # Number of episodes
env_generator = Environment(num_episodes)
episodes = env_generator.generate_episode()
