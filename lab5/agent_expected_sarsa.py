# agent_expected_sarsa.py

import sys
import gymnasium as gym
import numpy as np
import pickle
import matplotlib.pyplot as plt
from state_discretizer import StateDiscretizer

class ExpectedSARSAAgent:
    def __init__(self, iht_size=8*4096, alpha=0.1, gamma=0.99, epsilon=1.0, 
                 epsilon_min=0.01, epsilon_decay=0.995):
        """
        Initialize Expected SARSA agent.
        
        Args:
            iht_size: Size of the index hash table for tile coding
            alpha: Base learning rate (will be divided by num_tilings)
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for epsilon
        """
        # Initialize environment
        self.env = gym.make('LunarLander-v3', render_mode=None)  # Disable rendering for improved performance
        
        # Initialize state discretizer
        self.state_discretizer = StateDiscretizer(self.env, iht_size=iht_size)
        
        # Initialize Q-table (weights for each action)
        self.q_table = [np.zeros(self.state_discretizer.iht_size) 
                        for _ in range(4)]
        
        # Hyperparameters
        self.alpha = alpha / self.state_discretizer.num_tilings
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Training tracking
        self.best_average_reward = -float('inf')

    def select_action(self, state, testing=False):
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            testing: If True, use purely greedy policy (no exploration)
            
        Returns:
            int: Selected action (0-3)
        """
        state_features = self.state_discretizer.discretize(state)
        
        # During testing or exploitation, choose best action
        if testing or np.random.random() >= self.epsilon:
            q_values = [sum(self.q_table[a][tile] for tile in state_features) 
                       for a in range(4)]
            return np.argmax(q_values)
        else:
            # Exploration: random action
            return np.random.randint(4)

    def update(self, state, action, reward, next_state, done):
        """
        Expected SARSA update.
        Q(s,a) ← Q(s,a) + α[r + γ·Σ_a' π(a'|s')Q(s',a') - Q(s,a)]
        where the expectation is over the policy π
        
        Args:
            state: Previous state
            action: Action taken
            reward: Reward received
            next_state: New state
            done: Whether episode ended
        """
        state_features = self.state_discretizer.discretize(state)
        next_state_features = self.state_discretizer.discretize(next_state)
        
        # Current Q-value
        q_current = sum(self.q_table[action][tile] for tile in state_features)
        
        # Expected Q-value for next state under epsilon-greedy policy
        if done:
            expected_q_next = 0
        else:
            # Calculate Q-values for all actions in next state
            q_next_values = [sum(self.q_table[a][tile] for tile in next_state_features) 
                            for a in range(4)]
            
            # Find best action
            best_action = np.argmax(q_next_values)
            
            # Expected value under epsilon-greedy policy:
            # With probability (1-epsilon): take best action
            # With probability epsilon: take random action (uniform over 4 actions)
            expected_q_next = 0
            for a in range(4):
                if a == best_action:
                    # Probability of taking best action
                    prob = (1 - self.epsilon) + (self.epsilon / 4)
                else:
                    # Probability of taking non-best action
                    prob = self.epsilon / 4
                expected_q_next += prob * q_next_values[a]
        
        # TD error
        td_error = reward + self.gamma * expected_q_next - q_current
        
        # Update weights
        for tile in state_features:
            self.q_table[action][tile] += self.alpha * td_error

    def train(self, num_episodes):
        """
        Train the agent.
        
        Args:
            num_episodes: Number of episodes to train
        """
        rewards_history = []
        rolling_averages = []
        epsilon_history = []
        
        for episode in range(num_episodes):
            state, info = self.env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = self.select_action(state, testing=False)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Expected SARSA doesn't need next_action in update
                self.update(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
            
            rewards_history.append(total_reward)
            epsilon_history.append(self.epsilon)
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Calculate rolling average
            if len(rewards_history) >= 100:
                avg = np.mean(rewards_history[-100:])
            else:
                avg = np.mean(rewards_history)
            rolling_averages.append(avg)
            
            # Print progress every 100 episodes
            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}, Average Reward: {avg:.2f}, Epsilon: {self.epsilon:.3f}")
                
                # Save best model (only when epsilon is low)
                if avg > self.best_average_reward and self.epsilon < 0.1:
                    self.best_average_reward = avg
                    self.save_agent('expected_sarsa_best.pkl')
                    print(f"Best model saved! Average reward: {avg:.2f}")
        
        # Plot training results
        self._plot_training_results(rewards_history, rolling_averages, epsilon_history)
        
        return rewards_history, rolling_averages

    def test(self, num_episodes):
        """
        Test the agent with greedy policy.
        
        Args:
            num_episodes: Number of episodes to test
            
        Returns:
            float: Average reward over test episodes
        """
        test_rewards = []
        
        for episode in range(num_episodes):
            state, info = self.env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = self.select_action(state, testing=True)
                state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                total_reward += reward
            
            test_rewards.append(total_reward)
        
        avg_reward = np.mean(test_rewards)
        std_reward = np.std(test_rewards)
        
        print(f"\nTest Results (Expected SARSA) over {num_episodes} episodes:")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Std Dev: {std_reward:.2f}")
        print(f"Min: {np.min(test_rewards):.2f}")
        print(f"Max: {np.max(test_rewards):.2f}")
        
        return avg_reward

    def save_agent(self, file_name):
        """
        Save agent's Q-table and IHT dictionary.
        
        Args:
            file_name: File name to save to
        """
        save_data = {
            'q_table': self.q_table,
            'iht_dictionary': self.state_discretizer.iht.dictionary,
            'best_avg_reward': self.best_average_reward
        }
        with open(file_name, 'wb') as f:
            pickle.dump(save_data, f)

    def load_agent(self, file_name):
        """
        Load agent's Q-table and IHT dictionary.
        
        Args:
            file_name: File name to load from
        """
        with open(file_name, 'rb') as f:
            save_data = pickle.load(f)
        
        self.q_table = save_data['q_table']
        self.state_discretizer.iht.dictionary = save_data['iht_dictionary']
        self.best_average_reward = save_data.get('best_avg_reward', -float('inf'))

    def _plot_training_results(self, rewards, rolling_avg, epsilon):
        """Plot training results."""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        episodes = range(1, len(rewards) + 1)
        
        # Plot 1: Rewards and Rolling Average
        axes[0].plot(episodes, rewards, alpha=0.3, color='blue', label='Episode Reward')
        axes[0].plot(episodes, rolling_avg, color='red', linewidth=2, 
                    label='Rolling Avg (100 episodes)')
        axes[0].axhline(y=200, color='green', linestyle='--', label='Target (200)')
        axes[0].axhline(y=100, color='orange', linestyle='--', label='Min Required (100)')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Reward')
        axes[0].set_title('Expected SARSA: Training Progress')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Epsilon Decay
        axes[1].plot(episodes, epsilon, color='purple', linewidth=2)
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Epsilon (Exploration Rate)')
        axes[1].set_title('Epsilon Decay Over Time')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('expected_sarsa_training.png', dpi=300, bbox_inches='tight')
        print("\nTraining plot saved as 'expected_sarsa_training.png'")
        plt.close()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python agent_expected_sarsa.py <num_episodes>")
        sys.exit(1)
    
    num_training_episodes = int(sys.argv[1])
    
    # Train agent
    agent = ExpectedSARSAAgent(iht_size=8*4096, epsilon_min=0.050, epsilon_decay=0.9988, alpha=0.041)
    print(f"Training Expected SARSA agent for {num_training_episodes} episodes...")
    agent.train(num_training_episodes)
    print("Training completed.")
    
    # Test agent
    agent.test(100)
