import numpy as np
import random
from collections import namedtuple

# Enhancing Sarsd with a priority attribute
Sarsd = namedtuple('Sarsd', ('state', 'action', 'reward', 'new_state', 'done', 'priority'))

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.alpha = alpha  # Degree of prioritization
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self.position = 0

    def add_experience(self, sarsd):
        max_priority = max(self.priorities) if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(sarsd)
            self.priorities.append(max_priority)
        else:
            self.buffer[self.position] = sarsd
            self.priorities[self.position] = max_priority

        self.position = (self.position + 1) % self.capacity

    def update_priorities(self, indices, errors):
        """Update priorities based on TD errors."""
        for index, error in zip(indices, errors):
            self.priorities[index] = (np.abs(error) + 1e-5) ** self.alpha

    def sample_batch(self, batch_size, beta=0.4):
        total = len(self.buffer)
        priorities = np.array(self.priorities)
        probas = priorities ** self.alpha
        probas /= probas.sum()

        indices = np.random.choice(total, batch_size, p=probas)
        samples = [self.buffer[idx] for idx in indices]

        # Compute importance sampling weights
        weights = (total * probas[indices]) ** (-beta)
        weights /= weights.max()

        states, actions, rewards, new_states, dones = zip(
            *[(s.state, s.action, s.reward, s.new_state, s.done) for s in samples])

        return np.array(states), np.array(actions), np.array(rewards), np.array(new_states), np.array(dones), indices, weights

# Example usage:
buffer = PrioritizedReplayBuffer(capacity=1000)

# Add experiences (dummy data)
for _ in range(5):
    exp = Sarsd(state=np.zeros((84, 84, 4)), action=0, reward=0, new_state=np.zeros((84, 84, 4)), done=False, priority=0)
    buffer.add_experience(exp)

# Sample a batch
batch = buffer.sample_batch(batch_size=2)
print(batch)  # Returns states, actions, rewards, new_states, dones, indices, and weights

# Update priorities (dummy TD errors)
buffer.update_priorities(indices=[0, 1], errors=[0.1, 0.2])
