
import numpy as np
import pickle
import os

class QLearningAgent:

    def __init__(self, actions, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.q_table = {}
        self.actions = actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon

    def get_state_key(self, state):
        """Convert state list/tuple to a string key for the dictionary."""
        return str(state)

    def choose_action(self, state):
        """Choose action using epsilon-greedy policy."""
        state_key = self.get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(len(self.actions))
        
        # Epsilon-greedy: explore vs exploit
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.actions)  # Explore
        else:
            return self.actions[np.argmax(self.q_table[state_key])]  # Exploit best action

    def learn(self, state, action, reward, next_state):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(len(self.actions))
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(len(self.actions))
        predict = self.q_table[state_key][action]
        target = reward + self.gamma * np.max(self.q_table[next_state_key])
        self.q_table[state_key][action] += self.lr * (target - predict)

    def save_model(self, filename='q_table.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f'[SUCCESS] Model saved to {filename}')

    def load_model(self, filename='q_table.pkl'):
        """Load Q-table from file."""
        if os.path.exists(filename):
            try:
                with open(filename, 'rb') as f:
                    self.q_table = pickle.load(f)
                print(f'[SUCCESS] Model loaded from {filename}')
                return True
            except Exception as e:
                print(f'[ERROR] Error loading model: {e}')
                return False
        else:
            print(f'[WARNING] Model file not found: {filename}')
            return False