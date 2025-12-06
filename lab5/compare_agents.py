# compare_agents.py
"""
Script to compare the performance of all three RL agents.
"""

import numpy as np
import matplotlib.pyplot as plt
from agent_qlearning import QLearningAgent
from agent_sarsa import SARSAAgent
from agent_expected_sarsa import ExpectedSARSAAgent


def compare_agents_performance():
    """Load and test all three trained agents."""
    
    agents = {
        'Q-Learning': ('qlearning_best.pkl', QLearningAgent),
        'SARSA': ('sarsa_best148.pkl', SARSAAgent),
        'Expected SARSA': ('expected_sarsa_best.pkl', ExpectedSARSAAgent)
    }
    
    results = {}
    
    print("=" * 60)
    print("Testing All Agents")
    print("=" * 60)
    
    for name, (model_file, AgentClass) in agents.items():
        print(f"\n--- Testing {name} ---")
        try:
            agent = AgentClass()
            agent.load_agent(model_file)
            avg_reward = agent.test(100)
            results[name] = avg_reward
        except FileNotFoundError:
            print(f"Model file '{model_file}' not found. Train the agent first.")
            results[name] = None
        except Exception as e:
            print(f"Error testing {name}: {e}")
            results[name] = None
    
    # Plot comparison
    print("\n" + "=" * 60)
    print("Comparison Summary")
    print("=" * 60)
    
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if valid_results:
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        agents_list = list(valid_results.keys())
        scores = list(valid_results.values())
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c'][:len(agents_list)]
        
        bars = ax.bar(agents_list, scores, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.axhline(y=100, color='orange', linestyle='--', linewidth=2, 
                  label='Min Required (100)')
        ax.axhline(y=200, color='green', linestyle='--', linewidth=2, 
                  label='Target (200)')
        
        ax.set_ylabel('Average Reward (100 episodes)', fontsize=12)
        ax.set_title('Comparison of RL Agents on Lunar Lander', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('agents_comparison.png', dpi=300, bbox_inches='tight')
        print("\nComparison plot saved as 'agents_comparison.png'")
        plt.show()
        
        # Print summary table
        print("\nPerformance Summary:")
        print("-" * 40)
        for name, score in valid_results.items():
            status = "✓ PASS" if score >= 100 else "✗ FAIL"
            print(f"{name:20s}: {score:7.2f}  {status}")
        print("-" * 40)
        
        best_agent = max(valid_results, key=valid_results.get)
        print(f"\nBest Agent: {best_agent} ({valid_results[best_agent]:.2f})")
    else:
        print("\nNo valid results to compare. Please train the agents first.")


def train_all_agents(num_episodes):
    """Train all three agents with the same number of episodes."""
    
    agents = [
        ('Q-Learning', QLearningAgent),
        ('SARSA', SARSAAgent),
        ('Expected SARSA', ExpectedSARSAAgent)
    ]
    
    print("=" * 60)
    print(f"Training All Agents for {num_episodes} Episodes")
    print("=" * 60)
    
    for name, AgentClass in agents:
        print(f"\n{'=' * 60}")
        print(f"Training {name}")
        print('=' * 60)
        agent = AgentClass()
        agent.train(num_episodes)
        print(f"\n{name} training completed!")
        agent.test(100)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        # Training mode
        num_episodes = int(sys.argv[1])
        train_all_agents(num_episodes)
    else:
        # Comparison mode (assumes agents are already trained)
        compare_agents_performance()
