# complete_training_system.py
"""
Complete Training and Evaluation System
Advanced training pipeline with comprehensive evaluation and monitoring
"""

import os
import sys
import json
import torch
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from datetime import datetime
import pickle
from torch.utils.data import DataLoader

# Import our custom modules
from ui_action_system import (
    UIActionDataset, ActionPredictionModel, UIActionTrainer, 
    RealTimeExecutionAgent, train_model
)
from data_augmentation import DatasetEnhancer

class ExperimentTracker:
    """Track and log experiments"""
    
    def __init__(self, experiment_name: str, use_wandb: bool = False):
        self.experiment_name = experiment_name
        self.use_wandb = use_wandb
        self.metrics_history = []
        
        # Setup logging
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'logs/{experiment_name}_training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize wandb if requested
        if self.use_wandb:
            try:
                import wandb
                wandb.init(project="ui-action-prediction", name=experiment_name)
            except Exception as e:
                self.logger.warning(f"Failed to initialize wandb: {e}")
                self.use_wandb = False
    
    def log_metrics(self, metrics: Dict, step: int):
        """Log metrics to file and wandb"""
        metrics['step'] = step
        metrics['timestamp'] = datetime.now().isoformat()
        self.metrics_history.append(metrics)
        
        # Log to console
        self.logger.info(f"Step {step}: {metrics}")
        
        # Log to wandb
        if self.use_wandb:
            try:
                import wandb
                wandb.log(metrics, step=step)
            except:
                pass
    
    def save_metrics(self, filepath: str):
        """Save metrics history to file"""
        with open(filepath, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)

class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self, model, action_encoder, device='cuda'):
        self.model = model
        self.action_encoder = action_encoder
        self.device = device
    
    def evaluate_comprehensive(self, dataloader) -> Dict:
        """Comprehensive evaluation including per-class metrics"""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_coordinates_pred = []
        all_coordinates_true = []
        
        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(self.device)
                action_targets = batch['action_class'].squeeze().to(self.device)
                coord_targets = batch['coordinates'].to(self.device)
                
                action_logits, coord_pred = self.model(images)
                action_pred = torch.argmax(action_logits, dim=1)
                
                all_predictions.extend(action_pred.cpu().numpy())
                all_targets.extend(action_targets.cpu().numpy())
                all_coordinates_pred.extend(coord_pred.cpu().numpy())
                all_coordinates_true.extend(coord_targets.cpu().numpy())
        
        # Calculate metrics
        metrics = {}
        
        # Action classification metrics
        try:
            metrics['classification_report'] = classification_report(
                all_targets, all_predictions, 
                target_names=self.action_encoder.classes_,
                output_dict=True,
                zero_division=0
            )
        except Exception as e:
            print(f"Classification report failed: {e}")
            metrics['classification_report'] = {}
        
        # Coordinate regression metrics
        coord_pred = np.array(all_coordinates_pred)
        coord_true = np.array(all_coordinates_true)
        
        metrics['coordinate_mae'] = np.mean(np.abs(coord_pred - coord_true), axis=0).tolist()
        metrics['coordinate_rmse'] = np.sqrt(np.mean((coord_pred - coord_true)**2, axis=0)).tolist()
        
        # Pixel accuracy (within N pixels)
        pixel_thresholds = [5, 10, 20, 50]
        for threshold in pixel_thresholds:
            pixel_distances = np.sqrt(np.sum((coord_pred - coord_true)**2, axis=1))
            accuracy = np.mean(pixel_distances <= threshold/1920.0)  # Normalized
            metrics[f'pixel_accuracy_{threshold}px'] = accuracy
        
        return metrics
    
    def plot_confusion_matrix(self, dataloader, save_path: str):
        """Generate and save confusion matrix"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(self.device)
                action_targets = batch['action_class'].squeeze().to(self.device)
                
                action_logits, _ = self.model(images)
                action_pred = torch.argmax(action_logits, dim=1)
                
                all_predictions.extend(action_pred.cpu().numpy())
                all_targets.extend(action_targets.cpu().numpy())
        
        # Create confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', 
                   xticklabels=self.action_encoder.classes_,
                   yticklabels=self.action_encoder.classes_,
                   cmap='Blues')
        plt.title('Action Prediction Confusion Matrix')
        plt.ylabel('True Action')
        plt.xlabel('Predicted Action')
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_failure_cases(self, dataloader, num_samples: int = 20) -> List[Dict]:
        """Analyze failure cases for debugging"""
        self.model.eval()
        failure_cases = []
        
        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(self.device)
                action_targets = batch['action_class'].squeeze().to(self.device)
                coord_targets = batch['coordinates'].to(self.device)
                
                action_logits, coord_pred = self.model(images)
                action_pred = torch.argmax(action_logits, dim=1)
                
                # Find incorrect predictions
                incorrect_mask = action_pred != action_targets
                
                if incorrect_mask.any():
                    for i in torch.where(incorrect_mask)[0]:
                        if len(failure_cases) >= num_samples:
                            break
                        
                        failure_case = {
                            'predicted_action': self.action_encoder.classes_[action_pred[i]],
                            'true_action': self.action_encoder.classes_[action_targets[i]],
                            'predicted_coords': coord_pred[i].cpu().numpy().tolist(),
                            'true_coords': coord_targets[i].cpu().numpy().tolist(),
                            'confidence': torch.softmax(action_logits[i], dim=0).max().item()
                        }
                        failure_cases.append(failure_case)
                
                if len(failure_cases) >= num_samples:
                    break
        
        return failure_cases

class AdvancedTrainingPipeline:
    """Advanced training pipeline with all features"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize experiment tracker
        self.tracker = ExperimentTracker(
            config['experiment_name'], 
            config.get('use_wandb', False)
        )
        
        # Create output directories
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(exist_ok=True, parents=True)
        (self.output_dir / 'models').mkdir(exist_ok=True)
        (self.output_dir / 'plots').mkdir(exist_ok=True)
        (self.output_dir / 'logs').mkdir(exist_ok=True)
    
    def prepare_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare training, validation, and test data"""
        self.tracker.logger.info("Preparing dataset...")
        
        # Enhance dataset if requested
        if self.config.get('enhance_data', True):
            enhancer = DatasetEnhancer(str(self.output_dir / 'enhanced_data'))
            enhanced_files = enhancer.enhance_dataset(
                self.config['data_files'], 
                enhancement_factor=self.config.get('enhancement_factor', 5)
            )
            data_files = enhanced_files
        else:
            data_files = self.config['data_files']
        
        # Create dataset
        dataset = UIActionDataset(
            data_files, 
            augment=True
        )
        
        if len(dataset) == 0:
            raise ValueError("No valid data found in dataset!")
        
        # Split dataset
        total_size = len(dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        # Create dataloaders
        batch_size = self.config.get('batch_size', 8)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        self.action_encoder = dataset.action_encoder
        self.tracker.logger.info(f"Dataset prepared: {len(dataset)} total samples")
        self.tracker.logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return train_loader, val_loader, test_loader
    
    def create_model(self) -> ActionPredictionModel:
        """Create and initialize model"""
        num_classes = len(self.action_encoder.classes_)
        model = ActionPredictionModel(
            num_classes, 
            input_size=tuple(self.config.get('input_size', [224, 224]))
        )
        
        # Load pretrained weights if specified
        if self.config.get('pretrained_path'):
            checkpoint = torch.load(self.config['pretrained_path'])
            model.load_state_dict(checkpoint['model_state_dict'])
            self.tracker.logger.info(f"Loaded pretrained model from {self.config['pretrained_path']}")
        
        return model.to(self.device)
    
    def train(self):
        """Complete training pipeline"""
        self.tracker.logger.info("Starting training pipeline...")
        
        # Prepare data
        train_loader, val_loader, test_loader = self.prepare_data()
        
        # Create model
        model = self.create_model()
        
        # Create trainer
        trainer = UIActionTrainer(model, self.device)
        
        # Training parameters
        num_epochs = self.config.get('num_epochs', 50)
        early_stopping_patience = self.config.get('early_stopping_patience', 10)
        best_val_accuracy = 0
        patience_counter = 0
        
        # Training loop
        for epoch in range(num_epochs):
            self.tracker.logger.info(f"Epoch {epoch+1}/{num_epochs}")
            
            # Train
            train_metrics = trainer.train_epoch(train_loader)
            
            # Validate
            val_metrics = trainer.validate(val_loader)
            
            # Combine metrics
            combined_metrics = {
                'epoch': epoch + 1,
                'train_loss': train_metrics['total_loss'],
                'train_action_loss': train_metrics['action_loss'],
                'train_coord_loss': train_metrics['coord_loss'],
                'val_loss': val_metrics['total_loss'],
                'val_accuracy': val_metrics['action_accuracy'],
                'val_coord_mae': val_metrics['coord_mae']
            }
            
            # Log metrics
            self.tracker.log_metrics(combined_metrics, epoch + 1)
            
            # Save best model
            if val_metrics['action_accuracy'] > best_val_accuracy:
                best_val_accuracy = val_metrics['action_accuracy']
                patience_counter = 0
                
                # Save model
                model_path = self.output_dir / 'models' / 'best_model.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'val_accuracy': best_val_accuracy,
                    'config': self.config
                }, model_path)
                
                # Save action encoder
                encoder_path = self.output_dir / 'models' / 'action_encoder.pkl'
                with open(encoder_path, 'wb') as f:
                    pickle.dump(self.action_encoder, f)
                
                self.tracker.logger.info(f"Saved best model with accuracy: {best_val_accuracy:.4f}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                self.tracker.logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Final evaluation
        self.tracker.logger.info("Starting final evaluation...")
        evaluator = ModelEvaluator(model, self.action_encoder, self.device)
        
        # Comprehensive evaluation
        test_metrics = evaluator.evaluate_comprehensive(test_loader)
        
        # Plot confusion matrix
        evaluator.plot_confusion_matrix(
            test_loader, 
            str(self.output_dir / 'plots' / 'confusion_matrix.png')
        )
        
        # Analyze failure cases
        failure_cases = evaluator.analyze_failure_cases(test_loader)
        
        # Save evaluation results
        evaluation_results = {
            'test_metrics': test_metrics,
            'failure_cases': failure_cases,
            'final_config': self.config,
            'best_val_accuracy': best_val_accuracy
        }
        
        with open(self.output_dir / 'logs' / 'evaluation_results.json', 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        
        # Save metrics history
        self.tracker.save_metrics(str(self.output_dir / 'logs' / 'training_metrics.json'))
        
        self.tracker.logger.info("Training pipeline completed!")
        self.tracker.logger.info(f"Best validation accuracy: {best_val_accuracy:.4f}")
        self.tracker.logger.info(f"Results saved to: {self.output_dir}")
        
        return str(self.output_dir / 'models' / 'best_model.pth')

def create_config_template() -> Dict:
    """Create a configuration template"""
    return {
        'experiment_name': 'ui_action_experiment_1',
        'data_files': ['login_task_data.json'],  # List of your data files
        'output_dir': 'experiments/exp_1',
        'batch_size': 8,
        'num_epochs': 50,
        'early_stopping_patience': 10,
        'input_size': [224, 224],
        'enhance_data': True,
        'enhancement_factor': 5,
        'use_wandb': False,  # Set to True if you want to use Weights & Biases
        'pretrained_path': None  # Path to pretrained model if available
    }

def validate_data_format(data_file: str) -> bool:
    """Validate that data file has correct format"""
    try:
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        required_fields = ['task_label', 'screenshots', 'actions']
        for field in required_fields:
            if field not in data:
                print(f"Missing required field: {field}")
                return False
        
        # Validate screenshots
        if not data['screenshots']:
            print("No screenshots found")
            return False
            
        for screenshot in data['screenshots'][:1]:  # Check first screenshot
            if 'image_base64' not in screenshot:
                print("Screenshot missing image_base64")
                return False
        
        # Validate actions
        if not data['actions']:
            print("No actions found")
            return False
            
        for action in data['actions'][:1]:  # Check first action
            if 'type' not in action or 'action' not in action:
                print("Action missing required fields")
                return False
        
        print(f"Data file {data_file} is valid!")
        return True
        
    except Exception as e:
        print(f"Error validating {data_file}: {e}")
        return False

def demonstrate_execution(model_path: str, config: Dict):
    """Demonstrate the trained model in action"""
    try:
        # Initialize execution agent
        encoder_path = str(Path(model_path).parent / 'action_encoder.pkl')
        
        if not os.path.exists(encoder_path):
            print(f"Encoder file not found: {encoder_path}")
            return
            
        agent = RealTimeExecutionAgent(model_path, encoder_path)
        
        print("Real-time execution agent initialized!")
        print("The agent can now predict and execute actions based on screen content.")
        print("To run automation, call: agent.run_automation('task_name')")
        
        # Demo: Just predict without executing
        screenshot = agent.capture_screen()
        predicted_action = agent.predict_action(screenshot)
        
        print(f"Demo prediction from current screen:")
        print(f"  Action: {predicted_action['type']} - {predicted_action['action']}")
        print(f"  Coordinates: {predicted_action['coordinates']}")
        print(f"  Confidence: {predicted_action['confidence']:.3f}")
        
        if predicted_action['key']:
            print(f"  Key: {predicted_action['key']}")
            
    except Exception as e:
        print(f"Demo execution failed: {e}")

def setup_environment():
    """Setup the training environment"""
    print("Setting up UI Action Prediction environment...")
    
    # Create necessary directories
    directories = [
        'data', 'models', 'experiments', 'logs', 
        'enhanced_data', 'plots', 'configs'
    ]
    
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
    
    # Create sample config
    config = create_config_template()
    with open('configs/default_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create sample data structure
    sample_data = {
        "task_label": "login_example",
        "start_time": "2025-05-29T10:00:00.000000",
        "end_time": "2025-05-29T10:00:15.000000", 
        "duration_ms": 15000,
        "screen_size": {
            "width": 1920,
            "height": 1080
        },
        "screenshots": [
            {
                "timestamp_ms": 1000,
                "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
                "size": {"width": 1920, "height": 1080}
            }
        ],
        "actions": [
            {
                "timestamp_ms": 1200,
                "type": "mouse",
                "action": "click",
                "button": "left",
                "coordinates": {"x": 960, "y": 540}
            },
            {
                "timestamp_ms": 2000,
                "type": "keyboard", 
                "action": "press",
                "key": "u",
                "key_code": 117
            }
        ]
    }
    
    with open('data/sample_data_structure.json', 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print("Environment setup complete!")
    print("Created directories:", directories)
    print("Sample config: configs/default_config.json")
    print("Sample data structure: data/sample_data_structure.json")

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='UI Action Prediction Training')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--data-dir', type=str, help='Directory containing data files')
    parser.add_argument('--experiment-name', type=str, default='ui_action_exp', help='Experiment name')
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_config_template()
        
        # Update with command line arguments
        if args.data_dir:
            data_files = list(Path(args.data_dir).glob('*.json'))
            config['data_files'] = [str(f) for f in data_files]
        
        config['experiment_name'] = args.experiment_name
        
        # Save config template
        config_path = f"{config['experiment_name']}_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Created config template: {config_path}")
    
    # Validate data files exist
    if not config.get('data_files'):
        print("Error: No data files specified. Use --data-dir or provide config with data_files")
        return
    
    # Check if data files exist
    valid_files = []
    for file_path in config['data_files']:
        if os.path.exists(file_path):
            if validate_data_format(file_path):
                valid_files.append(file_path)
            else:
                print(f"Warning: Invalid data format in {file_path}")
        else:
            print(f"Warning: Data file not found: {file_path}")
    
    if not valid_files:
        print("Error: No valid data files found!")
        return
    
    config['data_files'] = valid_files
    print(f"Found {len(valid_files)} valid data files")
    
    # Run training pipeline
    try:
        pipeline = AdvancedTrainingPipeline(config)
        model_path = pipeline.train()
        
        print(f"\nTraining completed!")
        print(f"Best model saved to: {model_path}")
        
        # Demonstrate real-time execution
        print("\nStarting demonstration of trained model...")
        demonstrate_execution(model_path, config)
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Check if setup is needed
    if len(sys.argv) > 1 and sys.argv[1] == 'setup':
        setup_environment()
        sys.exit(0)
    
    # Run main training
    main()