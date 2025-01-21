# machine_learning.py

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import logging
import numpy as np
from collections import deque
import random
import asyncio

@dataclass
class RLConfig:
    """Configuration for reinforcement learning."""
    state_size: int
    action_size: int
    hidden_size: int = 128
    learning_rate: float = 0.001
    gamma: float = 0.99  # Discount factor
    epsilon: float = 1.0  # Exploration rate
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    memory_size: int = 10000
    batch_size: int = 64
    update_frequency: int = 10

class ReplayMemory:
    """Experience replay memory for reinforcement learning."""
    
    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state: torch.Tensor, action: int, reward: float, 
             next_state: torch.Tensor, done: bool):
        """Store transition in memory."""
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """Sample random batch from memory."""
        return random.sample(self.memory, batch_size)
    
    def __len__(self) -> int:
        return len(self.memory)

class RLAgent(nn.Module):
    """
    Reinforcement learning agent for optimizing processing pathways.
    Uses Deep Q-Network (DQN) architecture.
    """
    
    def __init__(self, config: RLConfig):
        super().__init__()
        self.config = config
        
        # Neural network layers
        self.fc1 = nn.Linear(config.state_size, config.hidden_size)
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.fc2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.ln2 = nn.LayerNorm(config.hidden_size)
        self.fc3 = nn.Linear(config.hidden_size, config.action_size)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            state: Input state tensor
            
        Returns:
            Q-values for each action
        """
        x = self.fc1(state)
        x = self.ln1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        return self.fc3(x)

class PathwayOptimizer:
    """
    Optimizes processing pathways using reinforcement learning.
    """
    
    def __init__(self, config: RLConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize networks
        self.policy_net = RLAgent(config).to(self.device)
        self.target_net = RLAgent(config).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.learning_rate)
        self.memory = ReplayMemory(config.memory_size)
        
        self.epsilon = config.epsilon
        self.steps_done = 0
        
        self.logger.info(f"Pathway Optimizer initialized on {self.device}")
    
    def select_action(self, state: torch.Tensor) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state tensor
            
        Returns:
            Selected action index
        """
        if random.random() > self.epsilon:
            with torch.no_grad():
                state = state.to(self.device)
                q_values = self.policy_net(state)
                return q_values.max(1)[1].item()
        else:
            return random.randrange(self.config.action_size)
    
    async def optimize_model(self) -> Optional[float]:
        """
        Perform one step of optimization on the model.
        
        Returns:
            Loss value if optimization was performed, None otherwise
        """
        if len(self.memory) < self.config.batch_size:
            return None
        
        # Sample transitions
        transitions = self.memory.sample(self.config.batch_size)
        batch = list(zip(*transitions))
        
        # Prepare batch
        state_batch = torch.stack(batch[0]).to(self.device)
        action_batch = torch.tensor(batch[1], device=self.device)
        reward_batch = torch.tensor(batch[2], device=self.device)
        next_state_batch = torch.stack(batch[3]).to(self.device)
        done_batch = torch.tensor(batch[4], device=self.device)
        
        # Compute Q(s_t, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # Compute V(s_{t+1}) for all next states
        next_state_values = torch.zeros(self.config.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values = self.target_net(next_state_batch).max(1)[0]
        next_state_values[done_batch] = 0.0
        
        # Compute expected Q values
        expected_state_action_values = (next_state_values * self.config.gamma) + reward_batch
        
        # Compute loss
        loss = nn.MSELoss()(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network weights with policy network weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def update_epsilon(self):
        """Update exploration rate."""
        self.epsilon = max(self.config.epsilon_min, 
                         self.epsilon * self.config.epsilon_decay)
    
    async def train_step(self, state: torch.Tensor, reward: float, done: bool) -> Tuple[int, float]:
        """
        Perform one training step.
        
        Args:
            state: Current state tensor
            reward: Reward value
            done: Whether episode is done
            
        Returns:
            Tuple of (selected action, loss value)
        """
        # Select action
        action = self.select_action(state)
        
        # Store transition in memory
        if len(self.memory) > 0:  # Only if we have a previous state
            self.memory.push(
                self.memory.memory[-1][3],  # Previous next_state becomes current state
                action,
                reward,
                state,
                done
            )
        
        # Perform optimization step
        loss = await self.optimize_model()
        
        # Update target network if needed
        if self.steps_done % self.config.update_frequency == 0:
            self.update_target_network()
        
        # Update exploration rate
        self.update_epsilon()
        
        self.steps_done += 1
        
        return action, loss if loss is not None else 0.0
    
    def get_state_features(self, metrics: Dict[str, float]) -> torch.Tensor:
        """
        Convert metrics dictionary to state tensor.
        
        Args:
            metrics: Dictionary of metric values
            
        Returns:
            State tensor
        """
        features = []
        for key in sorted(metrics.keys()):  # Sort to ensure consistent order
            features.append(metrics[key])
        
        # Pad if necessary
        while len(features) < self.config.state_size:
            features.append(0.0)
        
        # Truncate if necessary
        features = features[:self.config.state_size]
        
        return torch.tensor(features, dtype=torch.float32)
    
    def calculate_reward(self, metrics: Dict[str, float]) -> float:
        """
        Calculate reward based on performance metrics.
        
        Args:
            metrics: Dictionary of metric values
            
        Returns:
            Calculated reward value
        """
        # Example reward calculation using weighted sum of metrics
        reward = (
            0.3 * metrics.get('coherence', 0.0) +
            0.3 * metrics.get('stability', 0.0) +
            0.2 * metrics.get('integration', 0.0) +
            0.2 * metrics.get('efficiency', 0.0)
        )
        return float(reward)
    
    def save_model(self, path: str):
        """
        Save model state.
        
        Args:
            path: Path to save model
        """
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done
        }, path)
        self.logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """
        Load model state.
        
        Args:
            path: Path to load model from
        """
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        self.logger.info(f"Model loaded from {path}")

class MetricsTracker:
    """Track and analyze system performance metrics."""
    
    def __init__(self, window_size: int = 1000):
        self.metrics_history = deque(maxlen=window_size)
        self.logger = logging.getLogger(__name__)
    
    def update_metrics(self, metrics: Dict[str, float]):
        """
        Update metrics history.
        
        Args:
            metrics: Dictionary of metric values
        """
        self.metrics_history.append(metrics)
    
    def get_average_metrics(self, window: int = None) -> Dict[str, float]:
        """
        Get average metrics over specified window.
        
        Args:
            window: Optional window size (default: all history)
            
        Returns:
            Dictionary of averaged metrics
        """
        if not self.metrics_history:
            return {}
        
        if window is not None:
            history = list(self.metrics_history)[-window:]
        else:
            history = self.metrics_history
        
        avg_metrics = {}
        for key in history[0].keys():
            values = [m[key] for m in history]
            avg_metrics[key] = float(np.mean(values))
        
        return avg_metrics
    
    def get_metric_trends(self) -> Dict[str, float]:
        """
        Calculate metric trends using linear regression.
        
        Returns:
            Dictionary of metric trends (slopes)
        """
        if len(self.metrics_history) < 2:
            return {}
        
        trends = {}
        x = np.arange(len(self.metrics_history))
        
        for key in self.metrics_history[0].keys():
            y = [m[key] for m in self.metrics_history]
            slope = np.polyfit(x, y, 1)[0]
            trends[key] = float(slope)
        
        return trends
