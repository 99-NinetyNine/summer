# reward_learning_system.py
"""
Reward-Based Learning System for UI Action Automation
Implements reinforcement learning to improve task completion through rewards
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque, namedtuple
import random
import cv2
import time
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from pathlib import Path
import pyautogui
from PIL import Image
import io
import base64

from ui_action_system import ActionPredictionModel, RealTimeExecutionAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Experience:
    """Single experience for replay buffer"""
    state: np.ndarray  # Screenshot
    action: int  # Action taken
    reward: float  # Reward received
    next_state: np.ndarray  # Next screenshot
    done: bool  # Task completed/failed
    action_coords: Tuple[int, int]  # Mouse coordinates
    metadata: Dict  # Additional info

class TaskRewardCalculator:
    """Calculate rewards based on task progress and completion"""
    
    def __init__(self, task_type: str = "login"):
        self.task_type = task_type
        self.previous_state = None
        self.task_start_time = None
        self.action_count = 0
        self.setup_task_rewards()
    
    def setup_task_rewards(self):
        """Setup reward schemes for different task types"""
        self.reward_schemes = {
            'login': {
                'task_completion': 100.0,      # Successfully logged in
                'progress_step': 10.0,         # Each meaningful step (click field, type, etc.)
                'correct_element': 5.0,        # Clicked correct UI element
                'wrong_element': -5.0,         # Clicked wrong element
                'task_failure': -50.0,         # Task failed (error message, timeout)
                'action_penalty': -1.0,        # Small penalty per action (encourage efficiency)
                'time_bonus': 20.0,           # Bonus for completing quickly
                'accuracy_bonus': 15.0,        # Bonus for precise clicks
            },
            'search': {
                'task_completion': 80.0,
                'progress_step': 8.0,
                'correct_element': 4.0,
                'wrong_element': -4.0,
                'task_failure': -40.0,
                'action_penalty': -0.8,
                'time_bonus': 15.0,
                'accuracy_bonus': 12.0,
            },
            'form_fill': {
                'task_completion': 120.0,
                'progress_step': 12.0,
                'correct_element': 6.0,
                'wrong_element': -6.0,
                'task_failure': -60.0,
                'action_penalty': -1.2,
                'time_bonus': 25.0,
                'accuracy_bonus': 18.0,
            }
        }
        
        self.current_rewards = self.reward_schemes.get(self.task_type, self.reward_schemes['login'])
    
    def reset_task(self):
        """Reset for new task"""
        self.previous_state = None
        self.task_start_time = time.time()
        self.action_count = 0
    
    def calculate_reward(self, action_taken: Dict, current_state: np.ndarray, 
                        task_status: str, metadata: Dict = None) -> float:
        """Calculate reward for current action"""
        reward = 0.0
        self.action_count += 1
        
        # Base action penalty (encourage efficiency)
        reward += self.current_rewards['action_penalty']
        
        # Task completion rewards
        if task_status == 'completed':
            reward += self.current_rewards['task_completion']
            
            # Time bonus (complete faster = higher reward)
            if self.task_start_time:
                elapsed_time = time.time() - self.task_start_time
                if elapsed_time < 30:  # Under 30 seconds
                    reward += self.current_rewards['time_bonus']
                elif elapsed_time < 60:  # Under 1 minute
                    reward += self.current_rewards['time_bonus'] * 0.5
        
        elif task_status == 'failed':
            reward += self.current_rewards['task_failure']
        
        elif task_status == 'progress':
            reward += self.current_rewards['progress_step']
        
        # Element-specific rewards
        if metadata:
            element_type = metadata.get('element_clicked', '')
            if self._is_correct_element(element_type, action_taken):
                reward += self.current_rewards['correct_element']
            elif self._is_wrong_element(element_type, action_taken):
                reward += self.current_rewards['wrong_element']
            
            # Accuracy bonus for precise clicks
            if metadata.get('click_accuracy', 0) > 0.9:
                reward += self.current_rewards['accuracy_bonus']
        
        # Visual progress rewards
        visual_reward = self._calculate_visual_progress_reward(current_state)
        reward += visual_reward
        
        return reward
    
    def _is_correct_element(self, element_type: str, action: Dict) -> bool:
        """Check if clicked element is appropriate for the action"""
        if self.task_type == 'login':
            correct_elements = ['input', 'button', 'submit', 'login', 'email', 'password']
            return any(elem in element_type.lower() for elem in correct_elements)
        elif self.task_type == 'search':
            correct_elements = ['search', 'input', 'button', 'submit']
            return any(elem in element_type.lower() for elem in correct_elements)
        return False
    
    def _is_wrong_element(self, element_type: str, action: Dict) -> bool:
        """Check if clicked element is inappropriate"""
        wrong_elements = ['advertisement', 'social', 'unrelated', 'popup']
        return any(elem in element_type.lower() for elem in wrong_elements)
    
    def _calculate_visual_progress_reward(self, current_state: np.ndarray) -> float:
        """Calculate reward based on visual changes in the screen"""
        if self.previous_state is None:
            self.previous_state = current_state
            return 0.0
        
        # Calculate visual difference
        diff = cv2.absdiff(current_state, self.previous_state)
        change_ratio = np.sum(diff > 30) / (diff.shape[0] * diff.shape[1])
        
        # Reward meaningful visual changes
        if 0.01 < change_ratio < 0.3:  # Meaningful but not overwhelming change
            visual_reward = 2.0
        elif change_ratio > 0.3:  # Major change (possibly page load)
            visual_reward = 5.0
        else:
            visual_reward = 0.0
            
        self.previous_state = current_state
        return visual_reward

class TaskStateDetector:
    """Detect task completion/failure states from screenshots"""
    
    def __init__(self, task_type: str = "login"):
        self.task_type = task_type
        self.setup_detection_patterns()
    
    def setup_detection_patterns(self):
        """Setup detection patterns for different tasks"""
        self.success_indicators = {
            'login': [
                'dashboard', 'welcome', 'profile', 'logout', 'home',
                'signed in', 'account', 'settings'
            ],
            'search': [
                'results', 'found', 'matches', 'showing', 'items'
            ],
            'form_fill': [
                'submitted', 'thank you', 'confirmation', 'success',
                'received', 'processed'
            ]
        }
        
        self.failure_indicators = {
            'login': [
                'invalid', 'incorrect', 'error', 'failed', 'denied',
                'wrong password', 'user not found', 'try again'
            ],
            'search': [
                'no results', 'not found', 'error', 'failed'
            ],
            'form_fill': [
                'error', 'invalid', 'required', 'missing', 'failed'
            ]
        }
    
    def detect_task_status(self, screenshot: np.ndarray, ocr_text: str = None) -> str:
        """Detect current task status from screenshot"""
        
        # If OCR text is available, use text-based detection
        if ocr_text:
            return self._detect_from_text(ocr_text)
        
        # Otherwise use visual detection
        return self._detect_from_visual(screenshot)
    
    def _detect_from_text(self, text: str) -> str:
        """Detect status from OCR text"""
        text_lower = text.lower()
        
        # Check for success indicators
        success_patterns = self.success_indicators.get(self.task_type, [])
        for pattern in success_patterns:
            if pattern in text_lower:
                return 'completed'
        
        # Check for failure indicators
        failure_patterns = self.failure_indicators.get(self.task_type, [])
        for pattern in failure_patterns:
            if pattern in text_lower:
                return 'failed'
        
        return 'progress'
    
    def _detect_from_visual(self, screenshot: np.ndarray) -> str:
        """Detect status from visual cues"""
        # This is a simplified version - you could add more sophisticated detection
        # For now, we'll use basic heuristics
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY)
        
        # Look for common UI patterns
        # (This could be enhanced with template matching, etc.)
        
        return 'progress'  # Default to progress

class ReplayBuffer:
    """Experience replay buffer for training"""
    
    def __init__(self, capacity: int = 50000):
        self.buffer = deque(maxlen=capacity)
        self.Experience = namedtuple('Experience', 
                                   ['state', 'action', 'reward', 'next_state', 'done'])
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        experience = self.Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int):
        """Sample random batch of experiences"""
        experiences = random.sample(self.buffer, batch_size)
        
        states = torch.FloatTensor([e.state for e in experiences])
        actions = torch.LongTensor([e.action for e in experiences])
        rewards = torch.FloatTensor([e.reward for e in experiences])
        next_states = torch.FloatTensor([e.next_state for e in experiences])
        dones = torch.BoolTensor([e.done for e in experiences])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class RewardBasedUIAgent(nn.Module):
    """Enhanced UI agent with reward-based learning"""
    
    def __init__(self, base_model: ActionPredictionModel, action_space_size: int):
        super().__init__()
        self.base_model = base_model
        self.action_space_size = action_space_size
        
        # Add value head for RL
        feature_size = self.base_model.action_classifier[0].in_features
        self.value_head = nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # State value
        )
        
        # Add advantage head
        self.advantage_head = nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.ReLU(),
            nn.Linear(256, action_space_size)  # Action advantages
        )
    
    def forward(self, x):
        """Forward pass with dueling DQN architecture"""
        features = self.base_model.feature_extractor(x)
        features = features.view(features.size(0), -1)
        
        # Original action prediction and coordinates
        action_logits = self.base_model.action_classifier(features)
        coordinates = self.base_model.coordinate_regressor(features)
        
        # RL components
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        
        # Dueling DQN: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return action_logits, coordinates, q_values

class RewardBasedTrainer:
    """Trainer that incorporates rewards for continuous learning"""
    
    def __init__(self, agent: RewardBasedUIAgent, device: str = 'cuda'):
        self.agent = agent.to(device)
        self.target_agent = RewardBasedUIAgent(agent.base_model, agent.action_space_size).to(device)
        self.target_agent.load_state_dict(agent.state_dict())
        
        self.device = device
        self.optimizer = optim.Adam(agent.parameters(), lr=0.0001)
        self.replay_buffer = ReplayBuffer()
        
        # Training hyperparameters
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update_freq = 1000
        self.batch_size = 32
        
        self.training_step = 0
    
    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[int, Tuple[int, int]]:
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            # Random action (exploration)
            action_idx = random.randint(0, self.agent.action_space_size - 1)
            coordinates = (random.randint(0, 1920), random.randint(0, 1080))
        else:
            # Greedy action (exploitation)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action_logits, coord_pred, q_values = self.agent(state_tensor)
                
                action_idx = q_values.max(1)[1].item()
                coordinates = (
                    int(coord_pred[0][0].item() * 1920),
                    int(coord_pred[0][1].item() * 1080)
                )
        
        return action_idx, coordinates
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        # Preprocess states
        state = self._preprocess_state(state)
        next_state = self._preprocess_state(next_state)
        
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self):
        """Perform one training step"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Compute current Q values
        _, _, current_q_values = self.agent(states)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1))
        
        # Compute next Q values
        with torch.no_grad():
            _, _, next_q_values = self.target_agent(next_states)
            next_q_values = next_q_values.max(1)[0].detach()
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.target_agent.load_state_dict(self.agent.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def _preprocess_state(self, state: np.ndarray) -> np.ndarray:
        """Preprocess state for neural network"""
        # Resize and normalize
        state = cv2.resize(state, (224, 224))
        state = state.astype(np.float32) / 255.0
        state = np.transpose(state, (2, 0, 1))  # CHW format
        return state

class RewardBasedExecutionAgent(RealTimeExecutionAgent):
    """Enhanced execution agent with reward-based learning"""
    
    def __init__(self, model_path: str, action_encoder_path: str, 
                 task_type: str = 'login', device: str = 'cuda'):
        super().__init__(model_path, action_encoder_path, device)
        
        self.task_type = task_type
        self.reward_calculator = TaskRewardCalculator(task_type)
        self.state_detector = TaskStateDetector(task_type)
        
        # Convert to reward-based agent
        num_actions = len(self.action_encoder.classes_)
        self.reward_agent = RewardBasedUIAgent(self.model, num_actions)
        self.trainer = RewardBasedTrainer(self.reward_agent, device)
        
        # Learning tracking
        self.episode_rewards = []
        self.episode_actions = []
        self.current_episode_reward = 0
        
        # Auto-save best models
        self.best_reward = float('-inf')
        self.save_dir = Path("reward_models")
        self.save_dir.mkdir(exist_ok=True)
    
    def run_reward_based_automation(self, task_name: str, max_episodes: int = 100,
                                  max_actions_per_episode: int = 50):
        """Run automation with reward-based learning"""
        logger.info(f"Starting reward-based learning for task: {task_name}")
        
        for episode in range(max_episodes):
            logger.info(f"Episode {episode + 1}/{max_episodes}")
            
            episode_reward = self._run_single_episode(task_name, max_actions_per_episode)
            self.episode_rewards.append(episode_reward)
            
            # Log progress
            avg_reward = np.mean(self.episode_rewards[-10:])  # Last 10 episodes
            logger.info(f"Episode reward: {episode_reward:.2f}, Avg reward (10): {avg_reward:.2f}")
            
            # Save best model
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                self._save_best_model(episode, episode_reward)
            
            # Train the agent
            for _ in range(10):  # Multiple training steps per episode
                loss = self.trainer.train_step()
                if loss is not None:
                    logger.debug(f"Training loss: {loss:.4f}")
        
        # Save final results
        self._save_learning_results()
    
    def _run_single_episode(self, task_name: str, max_actions: int) -> float:
        """Run a single episode and return total reward"""
        self.reward_calculator.reset_task()
        self.current_episode_reward = 0
        episode_actions = []
        
        # Initial state
        current_state = self.capture_screen()
        
        for action_step in range(max_actions):
            # Select action
            action_idx, coordinates = self.trainer.select_action(current_state, training=True)
            
            # Convert to executable action
            action_str = self.action_encoder.inverse_transform([action_idx])[0]
            action_parts = action_str.split('_')
            
            executable_action = {
                'type': action_parts[0],
                'action': action_parts[1],
                'key': action_parts[2] if len(action_parts) > 2 else None,
                'coordinates': coordinates
            }
            
            # Execute action
            self.execute_action(executable_action)
            episode_actions.append(executable_action)
            
            # Get next state
            time.sleep(0.5)  # Wait for UI to update
            next_state = self.capture_screen()
            
            # Detect task status
            task_status = self.state_detector.detect_task_status(next_state)
            
            # Calculate reward
            reward = self.reward_calculator.calculate_reward(
                executable_action, next_state, task_status
            )
            
            self.current_episode_reward += reward
            
            # Store experience
            done = task_status in ['completed', 'failed']
            self.trainer.store_experience(
                current_state, action_idx, reward, next_state, done
            )
            
            # Update state
            current_state = next_state
            
            # End episode if task completed/failed
            if done:
                logger.info(f"Episode ended: {task_status} after {action_step + 1} actions")
                break
        
        return self.current_episode_reward
    
    def _save_best_model(self, episode: int, reward: float):
        """Save the best performing model"""
        save_path = self.save_dir / f"best_model_ep{episode}_reward{reward:.1f}.pth"
        torch.save({
            'episode': episode,
            'reward': reward,
            'model_state_dict': self.reward_agent.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict(),
            'epsilon': self.trainer.epsilon,
        }, save_path)
        logger.info(f"Saved new best model: {save_path}")
    
    def _save_learning_results(self):
        """Save learning progress and statistics"""
        results = {
            'episode_rewards': self.episode_rewards,
            'best_reward': self.best_reward,
            'task_type': self.task_type,
            'final_epsilon': self.trainer.epsilon,
            'total_episodes': len(self.episode_rewards),
        }
        
        results_path = self.save_dir / f"learning_results_{self.task_type}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Learning results saved: {results_path}")

# Usage example
def main():
    """Example usage of reward-based learning"""
    
    # Initialize reward-based agent
    agent = RewardBasedExecutionAgent(
        model_path="models/ui_action_model.pth",
        action_encoder_path="models/action_encoder.pkl",
        task_type="login"
    )
    
    # Run reward-based learning
    agent.run_reward_based_automation(
        task_name="login_training",
        max_episodes=50,
        max_actions_per_episode=30
    )
    
    print("Reward-based learning completed!")

if __name__ == "__main__":
    main()