import gym
import torch
import torch.nn as nn
import numpy as np
import pickle
import os
import pygame  # Required for event handling

# Defining the Q-network architecture for DQN
class QNetwork(nn.Module):
    def __init__(self, n_states, n_actions):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(n_states, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, n_actions)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Preprocess state (one-hot encoding)
def preprocess_state(state, n_states):
    state_tensor = torch.zeros(n_states)
    state_tensor[state] = 1  # One-hot encode the state
    return state_tensor

# Loading Q-table for SARSA or Q-learning
def load_q_table(path):
    if os.path.exists(path):
        with open(path, "rb") as file:
            Q = pickle.load(file)
        print(f"Q-table loaded from: {path}")
        return Q
    else:
        raise FileNotFoundError(f"No Q-table found at: {path}")

# Handling pygame events to prevent freezing
def handle_pygame_events():
    pygame.event.pump()  # Keeps the window responsive

# Play game using DQN
def play_with_dqn(env, model, n_states):
    state, info = env.reset()
    done = False
    total_reward = 0

    print("Playing with DQN model:")
    while not done:
        handle_pygame_events()  # Handling OS events
        env.render()
        action = model(preprocess_state(state, n_states)).argmax().item()  # Selecting the best action
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        print(f"Action: {action}, Reward: {reward}")
    
    print(f"Game Over. Total Reward: {total_reward}")

# Play game using SARSA or Q-learning Q-table
def play_with_q_table(env, Q_table):
    state, info = env.reset()
    done = False
    total_reward = 0

    print("Playing with Q-table model:")
    while not done:
        handle_pygame_events()  # Handling OS events
        env.render()
        action = np.argmax(Q_table[state])  # Selecting the best action from the Q-table
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        print(f"Action: {action}, Reward: {reward}")
    
    print(f"Game Over. Total Reward: {total_reward}")

# Main function to choose and evaluate the model
if __name__ == "__main__":
    # User-input parameters
    model_type = "Q-Learning"  # Options: "DQN", "SARSA", "Q-Learning"
    dqn_model_path = "dqn_taxi_episode_8000.pth"
    q_table_path = "taxi_q_table_episode_8000.pkl"  # For Q-learning (taxi_q_table_episode_8000.pkl) or SARSA (sarsa_model_episode_8000.pkl)

    # Initialize environment
    env = gym.make("Taxi-v3", render_mode="human")
    n_states = env.observation_space.n  # Total number of states
    n_actions = env.action_space.n  # Total number of actions

    try:
        if model_type == "DQN":
            # Load and evaluate DQN model
            model = QNetwork(n_states, n_actions)
            model.load_state_dict(torch.load(dqn_model_path))
            model.eval()
            play_with_dqn(env, model, n_states)
        
        elif model_type in ["SARSA", "Q-Learning"]:
            # Load and evaluate SARSA or Q-learning Q-table
            Q_table = load_q_table(q_table_path)
            play_with_q_table(env, Q_table)
        
        else:
            print("Invalid model type. Please choose 'DQN', 'SARSA', or 'Q-Learning'.")

    except FileNotFoundError as e:
        print(e)

    finally:
        env.close()
