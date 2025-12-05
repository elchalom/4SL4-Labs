# agent_template.py

import sys
import gymnasium as gym
import numpy as np
from state_discretizer import StateDiscretizer

class LunarLanderAgent:
    def __init__(self):
        """
        Initialize your agent here.

        This method is called when you create a new instance of your agent.
        Use this method to initialize variables, load models from files, etc.
        """
        # TODO: Initialize your agent's parameters and variables

        # Initialize environment
        self.env = gym.make('LunarLander-v3')
        
        self.num_actions = self.env.action_space.n

        # Initialize state discretizer if you are going to use Q-Learning
        self.state_discretizer = StateDiscretizer(self.env) #, iht_size=8192)

        # initialize Q-table or neural network weights
        self.q_table = [np.zeros(self.state_discretizer.iht_size) for _ in range(self.num_actions)]

        # Initiliaze Hyper Parameters
        self.alpha = 0.1 / 32       # Learning rate
        self.gamma = 0.99           # Discount factor
        self.epsilon = 1.0          # Exploration rate
        self.epsilon_min = 0.075     # Minimum exploration rate
        self.epsilon_decay = 0.995   # Decay rate for exploration probability

        # Initialize reward tracking
        self.best_average_reward = -float('inf')

    def select_action(self, state):
        """
        Select action using an epsilon-greedy policy.
        
        Otherwise, select the action with the highest Q-value for the current state.

        Args:
            state (array): The current state of the environment.

        Returns:
            int: The action to take.
        """
        
        # Discretize the state if you are going to use Q-Learning
        state_features = self.state_discretizer.discretize(state)
        
        # With Probability epsilon, select a random action.
        if np.random.random() < self.epsilon:
            # Explore: select a random action
            return np.random.randint(self.num_actions)
        
        # With probability (1 - epsilon), select the action with the highest Q-value (exploitation).
        q_values = []
        for action in range(self.num_actions):
            q_value = sum(self.q_table[action][tile] for tile in state_features)
            q_values.append(q_value)
            
        return np.argmax(q_values)

    def train(self, num_episodes):
        """
        Train your agent.

        Args:
            num_episodes (int): Number of episodes to train for.
        """
        rewards_history = []
        
        for episode in range(num_episodes):
            state, info = self.env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                self.update(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
                
            rewards_history.append(total_reward)
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            if (episode + 1) % 100 == 0:
                average_reward = np.mean(rewards_history[-100:])
                print(f"Episode {episode + 1}, Average Reward: {average_reward:.2f}, Epsilon: {self.epsilon:.3f}")
                
                # Save the best model based on average reward
                if average_reward > self.best_average_reward:
                    self.best_average_reward = average_reward
                    self.save_model('best_model.pkl')
                    print("Best model saved.")

    def update(self, state, action, reward, next_state, done):
        """
        Update your agent's knowledge based on the transition.
        Update Q-values using learning algorithm w/ tile coding.

        Args:
            state (array): The previous state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (array): The new state after the action.
            done (bool): Whether the episode has ended.
        """
    
        # Discretize the states if you are going to use Q-Learning
        state_features = self.state_discretizer.discretize(state)
        next_state_features = self.state_discretizer.discretize(next_state)

        # Calculate current Q-value Q(s, a)
        q_current = sum(self.q_table[action][tile] for tile in state_features)
        
        # Calculate the maximum Q-value for the next state max_a' Q(s', a')
        if done:
            q_next_max = 0
        else:
            # Calculate Q(s', a) for all 4 actions in next_state
            q_next_values = [
                sum(self.q_table[a][tile] for tile in next_state_features) 
                for a in range(self.num_actions)
            ]
            
            # Take the maximum over next actions
            q_next_max = max(q_next_values)
            
        # Calculate TD error
        td_error = reward + self.gamma * q_next_max - q_current
        
        # Update Q-value for all active tiles
        for tile in state_features:
            # Old weight + (learning rate * error)
            self.q_table[action][tile] += self.alpha * td_error
            

    def save_model(self, file_name):
        """
        Save your agent's model to a file.

        Args:
            file_name (str): The file name to save the model.
        """
        np.save(file_name, self.q_table)

    def load_model(self, file_name):
        """
        Load your agent's model from a file.

        Args:
            file_name (str): The file name to load the model from.
        """
        self.q_table = np.load(file_name, allow_pickle=True).tolist()

if __name__ == '__main__':

    agent = LunarLanderAgent()
    model_file = 'model.pkl'  # Set the model file name

    # Example usage:
    # Uncomment the following lines to train your agent and save the model

    num_training_episodes = int(sys.argv[1])  # Define the number of training episodes
    print("Training the agent...")
    agent.train(num_training_episodes)
    print("Training completed.")

    # Save the trained model
    # agent.save_model(model_file)
    # print("Model saved.")
