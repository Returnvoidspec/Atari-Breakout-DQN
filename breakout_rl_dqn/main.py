from collections import deque

from neural_network import Deep_neural_network  # The DQN model
from replay_buffer import ReplayBuffer  # To store and sample experiences
from train import train  # The training loop function
from pre_processing import process_and_stack_frames  # Frame preprocessing function
from config import *  # Import hyperparameters and other configurations
import numpy as np
import gymnasium as gym
import torch
from replay_buffer import Sarsd
import logging
from prioritized_replay_buffer import PrioritizedReplayBuffer  # Import the new buffer class
import pdb



def normal_train():
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting DQN training program.")

    # Initialize the environment and state
    logging.info("Initializing environment.")
    # Initialize the environment and state
    env_config = {'frameskip': 4}
    env = gym.make('ALE/Breakout-v5', render_mode='rgb_array', **env_config)# Custom wrapper around gym environment
    initial_state,_ = env.reset()


    # Initialize deque for stacked frames
    stacked_frames = deque([np.zeros((84, 84), dtype=int) for _ in range(AGENT_HISTORY_LENGTH)], maxlen=4)

    # Process and stack initial frame
    initial_state, stacked_frames = process_and_stack_frames(initial_state, stacked_frames, is_new_episode=True)

    # Initialize the DQN and target models
    num_actions = env.action_space.n
    dqn_model = Deep_neural_network(num_actions)
    target_model = Deep_neural_network(num_actions)
    target_model.load_state_dict(dqn_model.state_dict())  # Make sure both models have the same initial weights

    # Initialize the replay buffer
    replay_buffer = ReplayBuffer(capacity=REPLAY_MEMORY_SIZE)

    # Populate the initial replay buffer
    state = initial_state

    for _ in range(REPLAY_START_SIZE):
        action = np.random.choice(num_actions)  # Random initial actions
        next_state, reward, done, _,_ = env.step(action)

        # Stack the frames
        next_state, stacked_frames = process_and_stack_frames(next_state, stacked_frames, is_new_episode=False)

        replay_buffer.add_experience(Sarsd(state=state, action=action, reward=reward, new_state=next_state, done=done))

        if done:
            state,_ = env.reset()
            state, stacked_frames = process_and_stack_frames(state, stacked_frames, is_new_episode=True)
        else:
            state = next_state

    logging.info("Initial replay buffer populated.")


    # Start the training loop
    logging.info("Starting the training loop.")
    # Training loop
    train(dqn_model, target_model, env, replay_buffer)
    logging.info("DQN training program completed.")

def prioritze_train():
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting DQN training program.")

    # Initialize the environment and state
    logging.info("Initializing environment.")
    # Initialize the environment and state
    env_config = {'frameskip': 4}
    env = gym.make('ALE/Breakout-v5', render_mode='rgb_array', **env_config)# Custom wrapper around gym environment
    initial_state,_ = env.reset()


    # Initialize deque for stacked frames
    stacked_frames = deque([np.zeros((84, 84), dtype=int) for _ in range(AGENT_HISTORY_LENGTH)], maxlen=4)

    # Process and stack initial frame
    initial_state, stacked_frames = process_and_stack_frames(initial_state, stacked_frames, is_new_episode=True)

    # Initialize the DQN and target models
    num_actions = env.action_space.n
    dqn_model = Deep_neural_network(num_actions)
    target_model = Deep_neural_network(num_actions)
    target_model.load_state_dict(dqn_model.state_dict())  # Make sure both models have the same initial weights

    # Initialize the replay buffer
    replay_buffer = PrioritizedReplayBuffer(capacity=REPLAY_MEMORY_SIZE, alpha=PRIORITY_ALPHA)

    # Populate the initial replay buffer
    state = initial_state

    for _ in range(REPLAY_START_SIZE):
        action = np.random.choice(num_actions)
        next_state, reward, done, _, _ = env.step(action)
        next_state, stacked_frames = process_and_stack_frames(next_state, stacked_frames, is_new_episode=False)
        # Assign a high priority for initial experiences
        replay_buffer.add_experience(Sarsd(state=state, action=action, reward=reward, new_state=next_state, done=done),
                                     priority=1.0)

        if done:
            state,_ = env.reset()
            state, stacked_frames = process_and_stack_frames(state, stacked_frames, is_new_episode=True)
        else:
            state = next_state

    logging.info("Initial replay buffer populated.")


    # Start the training loop
    logging.info("Starting the training loop.")
    # Training loop
    train(dqn_model, target_model, env, replay_buffer)
    logging.info("DQN training program completed.")

def test():
    num_episodes = 10
    model_path = 'C:/Users/Martin/Desktop/Epita/Ing3/rl/breakout/Breakout-rl-DQN/breakout_rl_dqn/best_model/dqn_model_episode_10900.pth'
    # Load the environment
    env_config = {'frameskip': 4}
    env = gym.make('ALE/Breakout-v5', render_mode='human', **env_config)  # Render mode set to human for visualization

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

def main(flag=0):
    if flag == 0:
        test()
    if flag == 1:
        normal_train()
    else:
        prioritize_train()

if __name__ == "__main__":
    main()
