import gymnasium as gym
import torch
import numpy as np
from collections import deque
from breakout_rl_dqn.neural_network import Deep_neural_network  # Make sure to import your DQN model
from breakout_rl_dqn.pre_processing import process_and_stack_frames  # Frame preprocessing function


def test_model(model_path, num_episodes=10):
    # Load the environment
    env = gym.make('ALE/Breakout-v5', render_mode="human")  # Render mode set to human for visualization

    # Initialize the DQN model
    num_actions = env.action_space.n
    dqn_model = Deep_neural_network(num_actions)

    # Load the trained model weights
    dqn_model.load_state_dict(torch.load(model_path))
    dqn_model.eval()  # Set the model to evaluation mode

    for episode in range(num_episodes):
        state, _ = env.reset()

        # Initialize deque for stacked frames
        stacked_frames = deque([np.zeros((84, 84), dtype=int) for _ in range(4)], maxlen=4)
        state, stacked_frames = process_and_stack_frames(state, stacked_frames, is_new_episode=True)

        done = False
        episode_reward = 0

        while not done:
            # Choose an action based on the model's prediction
            state_tensor = torch.FloatTensor(state).unsqueeze(0).permute(0, 3, 1, 2)
            action = dqn_model(state_tensor).max(1)[1].view(1, 1).item()

            # Perform the action
            next_state, reward, done, _, _ = env.step(action)

            # Stack the frames
            next_state, stacked_frames = process_and_stack_frames(next_state, stacked_frames, is_new_episode=False)

            episode_reward += reward
            state = next_state

        print(f"Episode: {episode + 1}, Reward: {episode_reward}")

    env.close()


# Path to your saved model
model_path = 'C:/Users/Martin/Desktop/Epita/Ing3/rl/breakout/Breakout-rl-DQN/breakout_rl_dqn/best_model/dqn_model_episode_10900.pth'
test_model(model_path)
