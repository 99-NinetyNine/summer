# ui_action_system.py
"""
Main UI Action Prediction System
Contains the core model, dataset, trainer, and execution agent
"""

import json
import base64
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pyautogui
import time
import threading
from PIL import Image
import io
import random
from typing import List, Dict, Tuple, Optional
import albumentations as A
from sklearn.preprocessing import LabelEncoder
import pickle
import os
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ActionData:
    """Data structure for UI actions"""
    type: str  # 'mouse' or 'keyboard'
    action: str  # 'click', 'type', 'scroll', etc.
    coordinates: Optional[Tuple[int, int]] = None
    key: Optional[str] = None
    text: Optional[str] = None
    button: Optional[str] = None

class UIActionDataset(Dataset):
    """Dataset for UI action prediction"""
    
    def __init__(self, data_files: List[str], transform=None, augment=True):
        self.data = []
        self.transform = transform
        self.augment = augment
        self.action_encoder = LabelEncoder()
        
        # Load and preprocess data
        self._load_data(data_files)
        self._prepare_actions()
        
    def _load_data(self, data_files: List[str]):
        """Load data from JSON files"""
        all_sequences = []
        
        for file_path in data_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    all_sequences.append(data)
                logger.info(f"Loaded data from {file_path}")
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
        
        # Process sequences
        for seq in all_sequences:
            self._process_sequence(seq)
    
    def _process_sequence(self, sequence_data: Dict):
        """Process a single sequence of actions"""
        screenshots = sequence_data.get('screenshots', [])
        actions = sequence_data.get('actions', [])
        task_label = sequence_data.get('task_label', 'unknown')
        
        if not screenshots or not actions:
            logger.warning(f"Empty sequence found for task: {task_label}")
            return
        
        # Decode screenshots
        decoded_screenshots = []
        for screenshot in screenshots:
            try:
                img_data = base64.b64decode(screenshot['image_base64'])
                img = Image.open(io.BytesIO(img_data))
                img_array = np.array(img)
                decoded_screenshots.append({
                    'timestamp': screenshot['timestamp_ms'],
                    'image': img_array,
                    'size': screenshot['size']
                })
            except Exception as e:
                logger.error(f"Failed to decode screenshot: {e}")
        
        # Create training pairs (screenshot, next_action)
        self._create_training_pairs(decoded_screenshots, actions, task_label)
    
    def _create_training_pairs(self, screenshots: List[Dict], actions: List[Dict], task_label: str):
        """Create training pairs from screenshots and actions"""
        # Sort by timestamp
        screenshots.sort(key=lambda x: x['timestamp'])
        actions.sort(key=lambda x: x['timestamp_ms'])
        
        for i, action in enumerate(actions):
            # Find the screenshot closest to this action
            closest_screenshot = self._find_closest_screenshot(
                action['timestamp_ms'], screenshots
            )
            
            if closest_screenshot is not None:
                # Create action representation
                action_data = self._encode_action(action)
                
                training_pair = {
                    'image': closest_screenshot['image'],
                    'action': action_data,
                    'task_label': task_label,
                    'timestamp': action['timestamp_ms']
                }
                
                self.data.append(training_pair)
    
    def _find_closest_screenshot(self, action_timestamp: int, screenshots: List[Dict]) -> Optional[Dict]:
        """Find screenshot closest to action timestamp"""
        if not screenshots:
            return None
        
        closest = min(screenshots, key=lambda x: abs(x['timestamp'] - action_timestamp))
        # Only use if within reasonable time window (e.g., 2 seconds)
        if abs(closest['timestamp'] - action_timestamp) <= 2000:
            return closest
        return None
    
    def _encode_action(self, action: Dict) -> Dict:
        """Encode action into standardized format"""
        action_type = action.get('type', 'unknown')
        action_name = action.get('action', 'unknown')
        
        encoded = {
            'type': action_type,
            'action': action_name,
            'coordinates': action.get('coordinates', {'x': 0, 'y': 0}),
            'key': action.get('key', ''),
            'button': action.get('button', ''),
            'scroll': action.get('scroll', {'dx': 0, 'dy': 0})
        }
        
        return encoded
    
    def _prepare_actions(self):
        """Prepare action encodings"""
        # Extract all unique action combinations
        action_strings = []
        for item in self.data:
            action = item['action']
            action_str = f"{action['type']}_{action['action']}"
            if action['key']:
                action_str += f"_{action['key']}"
            action_strings.append(action_str)
        
        if not action_strings:
            logger.error("No valid actions found in dataset!")
            return
        
        # Fit label encoder
        self.action_encoder.fit(action_strings)
        
        # Add encoded actions to data
        for item in self.data:
            action = item['action']
            action_str = f"{action['type']}_{action['action']}"
            if action['key']:
                action_str += f"_{action['key']}"
            item['action_encoded'] = self.action_encoder.transform([action_str])[0]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image = item['image']
        action = item['action']
        
        # Apply augmentations
        if self.augment and self.transform:
            try:
                augmented = self.transform(image=image)
                image = augmented['image']
            except Exception as e:
                logger.warning(f"Augmentation failed: {e}")
        
        # Convert to tensor
        image = torch.FloatTensor(image).permute(2, 0, 1) / 255.0
        
        # Create target tensors
        action_class = torch.LongTensor([item['action_encoded']])
        coordinates = torch.FloatTensor([
            action['coordinates']['x'] / 1920.0,  # Normalize coordinates
            action['coordinates']['y'] / 1080.0
        ])
        
        return {
            'image': image,
            'action_class': action_class,
            'coordinates': coordinates,
            'raw_action': action
        }

class ActionPredictionModel(nn.Module):
    """CNN model for predicting UI actions"""
    
    def __init__(self, num_action_classes: int, input_size: Tuple[int, int] = (224, 224)):
        super().__init__()
        self.input_size = input_size
        self.num_action_classes = num_action_classes
        
        # CNN backbone
        self.feature_extractor = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # Calculate feature size
        feature_size = self._get_feature_size()
        
        # Action classification head
        self.action_classifier = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_action_classes)
        )
        
        # Coordinate regression head
        self.coordinate_regressor = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # x, y coordinates
        )
    
    def _get_feature_size(self):
        """Calculate the size of features after CNN"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, *self.input_size)
            features = self.feature_extractor(dummy_input)
            return features.view(1, -1).size(1)
    
    def forward(self, x):
        # Resize input if needed
        if x.size(-2) != self.input_size[0] or x.size(-1) != self.input_size[1]:
            x = torch.nn.functional.interpolate(x, size=self.input_size, mode='bilinear', align_corners=False)
        
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        
        action_logits = self.action_classifier(features)
        coordinates = self.coordinate_regressor(features)
        
        return action_logits, coordinates

class UIActionTrainer:
    """Trainer for the action prediction model"""
    
    def __init__(self, model: ActionPredictionModel, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.action_criterion = nn.CrossEntropyLoss()
        self.coord_criterion = nn.MSELoss()
        
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        action_loss_sum = 0
        coord_loss_sum = 0
        num_batches = 0
        
        for batch in dataloader:
            images = batch['image'].to(self.device)
            action_targets = batch['action_class'].squeeze().to(self.device)
            coord_targets = batch['coordinates'].to(self.device)
            
            self.optimizer.zero_grad()
            
            action_logits, coord_pred = self.model(images)
            
            action_loss = self.action_criterion(action_logits, action_targets)
            coord_loss = self.coord_criterion(coord_pred, coord_targets)
            
            # Weighted combination
            total_batch_loss = action_loss + 0.5 * coord_loss
            
            total_batch_loss.backward()
            self.optimizer.step()
            
            total_loss += total_batch_loss.item()
            action_loss_sum += action_loss.item()
            coord_loss_sum += coord_loss.item()
            num_batches += 1
        
        return {
            'total_loss': total_loss / num_batches,
            'action_loss': action_loss_sum / num_batches,
            'coord_loss': coord_loss_sum / num_batches
        }
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        action_correct = 0
        total_samples = 0
        coord_error = 0
        
        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(self.device)
                action_targets = batch['action_class'].squeeze().to(self.device)
                coord_targets = batch['coordinates'].to(self.device)
                
                action_logits, coord_pred = self.model(images)
                
                action_loss = self.action_criterion(action_logits, action_targets)
                coord_loss = self.coord_criterion(coord_pred, coord_targets)
                total_loss += (action_loss + 0.5 * coord_loss).item()
                
                # Calculate accuracy
                action_pred = torch.argmax(action_logits, dim=1)
                action_correct += (action_pred == action_targets).sum().item()
                total_samples += action_targets.size(0)
                
                # Calculate coordinate error
                coord_error += torch.mean(torch.abs(coord_pred - coord_targets)).item()
        
        return {
            'total_loss': total_loss / len(dataloader),
            'action_accuracy': action_correct / total_samples,
            'coord_mae': coord_error / len(dataloader)
        }

class RealTimeExecutionAgent:
    """Real-time execution agent for UI automation"""
    
    def __init__(self, model_path: str, action_encoder_path: str, device: str = 'cuda'):
        self.device = device
        self.running = False
        
        # Load model and encoder
        self.action_encoder = self._load_encoder(action_encoder_path)
        self.model = self._load_model(model_path)
        
        # Configure pyautogui
        pyautogui.PAUSE = 0.1
        pyautogui.FAILSAFE = True
        
    def _load_model(self, model_path: str) -> ActionPredictionModel:
        """Load trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        num_classes = len(self.action_encoder.classes_)
        
        model = ActionPredictionModel(num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model
    
    def _load_encoder(self, encoder_path: str) -> LabelEncoder:
        """Load action encoder"""
        with open(encoder_path, 'rb') as f:
            return pickle.load(f)
    
    def capture_screen(self) -> np.ndarray:
        """Capture current screen"""
        screenshot = pyautogui.screenshot()
        return np.array(screenshot)
    
    def predict_action(self, screenshot: np.ndarray) -> Dict:
        """Predict next action from screenshot"""
        # Preprocess image
        image = torch.FloatTensor(screenshot).permute(2, 0, 1).unsqueeze(0) / 255.0
        image = image.to(self.device)
        
        with torch.no_grad():
            action_logits, coordinates = self.model(image)
            
            # Get predictions
            action_class = torch.argmax(action_logits, dim=1).cpu().numpy()[0]
            coords = coordinates.cpu().numpy()[0]
            
            # Denormalize coordinates
            x = int(coords[0] * 1920)
            y = int(coords[1] * 1080)
            
            # Decode action
            action_str = self.action_encoder.inverse_transform([action_class])[0]
            action_parts = action_str.split('_')
            
            return {
                'type': action_parts[0],
                'action': action_parts[1],
                'key': action_parts[2] if len(action_parts) > 2 else None,
                'coordinates': (x, y),
                'confidence': torch.softmax(action_logits, dim=1).max().item()
            }
    
    def execute_action(self, action: Dict):
        """Execute predicted action"""
        try:
            if action['type'] == 'mouse':
                if action['action'] == 'click':
                    pyautogui.click(action['coordinates'][0], action['coordinates'][1])
                elif action['action'] == 'scroll':
                    pyautogui.scroll(1 if action.get('scroll_up', True) else -1,
                                   x=action['coordinates'][0], y=action['coordinates'][1])
            
            elif action['type'] == 'keyboard':
                if action['action'] == 'press' and action['key']:
                    if action['key'] == 'tab':
                        pyautogui.press('tab')
                    elif action['key'] == 'backspace':
                        pyautogui.press('backspace')
                    else:
                        pyautogui.press(action['key'])
            
            logger.info(f"Executed action: {action}")
            
        except Exception as e:
            logger.error(f"Failed to execute action {action}: {e}")
    
    def run_automation(self, task_name: str, max_actions: int = 50, confidence_threshold: float = 0.5):
        """Run automated task execution"""
        logger.info(f"Starting automation for task: {task_name}")
        self.running = True
        action_count = 0
        
        try:
            while self.running and action_count < max_actions:
                # Capture screen
                screenshot = self.capture_screen()
                
                # Predict action
                predicted_action = self.predict_action(screenshot)
                
                # Check confidence
                if predicted_action['confidence'] < confidence_threshold:
                    logger.warning(f"Low confidence action: {predicted_action['confidence']:.3f}")
                    time.sleep(1)
                    continue
                
                # Execute action
                self.execute_action(predicted_action)
                action_count += 1
                
                # Wait before next action
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            logger.info("Automation stopped by user")
        finally:
            self.running = False
    
    def stop(self):
        """Stop the automation"""
        self.running = False

def train_model(data_files: List[str], model_save_path: str, encoder_save_path: str):
    """Simple training function"""
    logger.info("Starting model training...")
    
    # Create dataset
    dataset = UIActionDataset(data_files, augment=True)
    
    if len(dataset) == 0:
        logger.error("No data found in dataset!")
        return
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # Save encoder
    with open(encoder_save_path, 'wb') as f:
        pickle.dump(dataset.action_encoder, f)
    
    # Create model
    num_classes = len(dataset.action_encoder.classes_)
    model = ActionPredictionModel(num_classes)
    
    # Create trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = UIActionTrainer(model, device)
    
    # Training loop
    best_accuracy = 0
    for epoch in range(50):
        train_metrics = trainer.train_epoch(train_loader)
        val_metrics = trainer.validate(val_loader)
        
        logger.info(f"Epoch {epoch+1}/50:")
        logger.info(f"  Train Loss: {train_metrics['total_loss']:.4f}")
        logger.info(f"  Val Loss: {val_metrics['total_loss']:.4f}")
        logger.info(f"  Val Accuracy: {val_metrics['action_accuracy']:.4f}")
        
        # Save best model
        if val_metrics['action_accuracy'] > best_accuracy:
            best_accuracy = val_metrics['action_accuracy']
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'epoch': epoch,
                'accuracy': best_accuracy
            }, model_save_path)
            logger.info(f"Saved new best model with accuracy: {best_accuracy:.4f}")

if __name__ == "__main__":
    # Example usage
    data_files = ["data/episode_login_20250529_095549.json"]  # Add your data files
    model_path = "ui_action_model.pth"
    encoder_path = "action_encoder.pkl"
    
    # Train the model
    train_model(data_files, model_path, encoder_path)