## training code
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

class GUIActionDataset(Dataset):
    """Dataset class for GUI action sequences"""
    
    def __init__(self, action_sequences, images, labels, label_encoder=None):
        self.action_sequences = torch.FloatTensor(action_sequences)
        self.images = torch.FloatTensor(images).permute(0, 3, 1, 2)  # Convert to CHW format
        
        if label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.labels = torch.LongTensor(self.label_encoder.fit_transform(labels))
        else:
            self.label_encoder = label_encoder
            self.labels = torch.LongTensor(self.label_encoder.transform(labels))
    
    def __len__(self):
        return len(self.action_sequences)
    
    def __getitem__(self, idx):
        return {
            'actions': self.action_sequences[idx],
            'image': self.images[idx],
            'label': self.labels[idx]
        }

class VisionEncoder(nn.Module):
    """CNN encoder for screenshot images"""
    
    def __init__(self, output_dim=512):
        super(VisionEncoder, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, output_dim)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ActionEncoder(nn.Module):
    """LSTM encoder for action sequences"""
    
    def __init__(self, input_dim=13, hidden_dim=256, num_layers=2, output_dim=512):
        super(ActionEncoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=0.3, bidirectional=True)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, output_dim)
        )
    
    def forward(self, x):
        # x shape: (batch, sequence_length, features)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last output of the LSTM
        last_output = lstm_out[:, -1, :]  # Get last timestep
        output = self.fc(last_output)
        return output

class MultimodalGUIModel(nn.Module):
    """Multimodal model combining vision and action sequence"""
    
    def __init__(self, num_classes, action_input_dim=13):
        super(MultimodalGUIModel, self).__init__()
        
        self.vision_encoder = VisionEncoder(output_dim=512)
        self.action_encoder = ActionEncoder(input_dim=action_input_dim, output_dim=512)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(1024, 512),  # 512 + 512
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Classification head
        self.classifier = nn.Linear(256, num_classes)
        
        # Action prediction head (for next action prediction)
        self.action_predictor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_input_dim)  # Predict next action features
        )
    
    def forward(self, images, actions):
        # Encode vision and actions
        vision_features = self.vision_encoder(images)
        action_features = self.action_encoder(actions)
        
        # Fuse features
        combined = torch.cat([vision_features, action_features], dim=1)
        fused_features = self.fusion(combined)
        
        # Task classification
        task_logits = self.classifier(fused_features)
        
        # Next action prediction
        next_action = self.action_predictor(fused_features)
        
        return task_logits, next_action, fused_features

class GUIModelTrainer:
    """Training pipeline for GUI automation model"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5)
        
        self.classification_loss = nn.CrossEntropyLoss()
        self.regression_loss = nn.MSELoss()
        
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(dataloader, desc="Training"):
            images = batch['image'].to(self.device)
            actions = batch['actions'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            task_logits, next_action_pred, _ = self.model(images, actions)
            
            # Classification loss
            class_loss = self.classification_loss(task_logits, labels)
            
            # Next action prediction loss (using last action as target)
            target_action = actions[:, -1, :]  # Last action in sequence
            action_loss = self.regression_loss(next_action_pred, target_action)
            
            # Combined loss
            total_batch_loss = class_loss + 0.5 * action_loss
            
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += total_batch_loss.item()
        
        return total_loss / len(dataloader)
    
    def validate_epoch(self, dataloader):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                images = batch['image'].to(self.device)
                actions = batch['actions'].to(self.device)
                labels = batch['label'].to(self.device)
                
                task_logits, next_action_pred, _ = self.model(images, actions)
                
                # Classification loss
                class_loss = self.classification_loss(task_logits, labels)
                
                # Next action prediction loss
                target_action = actions[:, -1, :]
                action_loss = self.regression_loss(next_action_pred, target_action)
                
                total_batch_loss = class_loss + 0.5 * action_loss
                total_loss += total_batch_loss.item()
                
                # Accuracy
                _, predicted = torch.max(task_logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(dataloader)
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, epochs=50):
        """Full training loop"""
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Training
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss, val_accuracy = self.validate_epoch(val_loader)
            self.val_losses.append(val_loss)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy
                }, 'best_gui_model.pth')
                print("Saved best model!")
        
        self.plot_training_curves()
    
    def plot_training_curves(self):
        """Plot training and validation loss curves"""
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss Curves')
        
        plt.tight_layout()
        plt.savefig('training_curves.png')
        plt.show()
