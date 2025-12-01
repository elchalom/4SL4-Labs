# agent_template.py

import gymnasium as gym
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

        # Initialize state discretizer if you are going to use Q-Learning
        # self.state_discretizer = StateDiscretizer(self.env)

        # initialize Q-table or neural network weights
        # self.q_table = [np.zeros(self.state_discretizer.iht_size) for _ in range(self.num_actions)]

        # Initialize any other parameters and variables
        # ...

        pass

    def select_action(self, state):
        """
        Given a state, select an action to take.

        Args:
            state (array): The current state of the environment.

        Returns:
            int: The action to take.
        """
        # TODO: Implement your action selection policy here
        # For example, you might use an epsilon-greedy policy if you're using Q-learning
        # Ensure the action returned is an integer in the range [0, 3]
         
        # Discretize the state if you are going to use Q-Learning
        # state_features = self.state_discretizer.discretize(state)
        
        pass

    def train(self, num_episodes):
        """
        Train your agent.

        Args:
            num_episodes (int): Number of episodes to train for.
        """
        # TODO: Implement your training loop here
        # Make sure to:
        # 1) Evaluate the trainign in each episode by monitoring the averge of the previous ~100
        #    episodes cumulative rewards.
        # 2) Autosave the best model achived in each epoch based on the evaluation.
        pass

    def update(self, state, action, reward, next_state, done):
        """
        Update your agent's knowledge based on the transition.

        Args:
            state (array): The previous state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (array): The new state after the action.
            done (bool): Whether the episode has ended.
        """
        # TODO: Implement your agent's update logic here
        # This method is where you would update your Q-table or neural network

        # Discretize the states if you are going to use Q-Learning
        # state_features = self.state_discretizer.discretize(state)
        # next_state_features = self.state_discretizer.discretize(next_state)

        pass

    def save_model(self, file_name):
        """
        Save your agent's model to a file.

        Args:
            file_name (str): The file name to save the model.
        """
        # TODO: Implement code to save your model (e.g., Q-table, neural network weights)
        # Example: np.save(file_name, self.q_table)
        pass

    def load_model(self, file_name):
        """
        Load your agent's model from a file.

        Args:
            file_name (str): The file name to load the model from.
        """
        # TODO: Implement code to load your model
        # Example: self.q_table = np.load(file_name)
        pass

if __name__ == '__main__':

    agent = LunarLanderAgent()
    model_file = 'model.pkl'  # Set the model file name

    # Example usage:
    # Uncomment the following lines to train your agent and save the model

    # num_training_episodes = 1000  # Define the number of training episodes
    # print("Training the agent...")
    # agent.train(num_training_episodes)
    # print("Training completed.")

    # Save the trained model
    # agent.save_model(model_file)
    # print("Model saved.")
