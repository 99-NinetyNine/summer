import torch
import torch.nn.functional as F
import numpy as np
import cv2
import pyautogui
import pynput
from pynput import mouse, keyboard
import time
import json
import pickle
from PIL import Image, ImageGrab
import threading
import queue
from typing import List, Dict, Tuple, Optional
import logging

# Import the model architecture
from NN_model import MultimodalGUIModel, VisionEncoder, ActionEncoder

class GUIAgent:
    """AI Agent for executing GUI automation tasks"""
    
    def __init__(self, model_path: str, label_encoder_path: str, 
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.screen_width = 1920
        self.screen_height = 1080
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load model and label encoder
        self.load_model(model_path, label_encoder_path)
        
        # Initialize GUI control
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.1
        
        # Action history for context
        self.action_history = []
        self.max_history_length = 100
        
        # Execution state
        self.is_executing = False
        self.execution_thread = None
        
        
    def load_model(self, model_path: str, label_encoder_path: str):
        """Load trained model and label encoder"""
        # Load label encoder
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        num_classes = len(self.label_encoder.classes_)
        
        # Initialize and load model
        self.model = MultimodalGUIModel(num_classes=num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        self.logger.info(f"Model loaded with {num_classes} task classes")
    
    def capture_screen(self) -> np.ndarray:
        """Capture current screen state"""
        screenshot = ImageGrab.grab()
        screenshot = np.array(screenshot)
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
        
        # Resize to model input size
        screenshot_resized = cv2.resize(screenshot, (224, 224))
        return screenshot_resized / 255.0
    
    def create_action_embedding(self, action: Dict) -> np.ndarray:
        """Convert action to numerical embedding"""
        embedding = [
            action.get('timestamp', 0) / 10000,
            action.get('time_delta', 0) / 1000,
            action.get('x_norm', 0),
            action.get('y_norm', 0),
            1 if action.get('type') == 'mouse' else 0,
            1 if action.get('type') == 'keyboard' else 0,
            1 if action.get('action') == 'click' else 0,
            1 if action.get('action') == 'press' else 0,
            1 if action.get('action') == 'release' else 0,
            1 if action.get('action') == 'scroll' else 0,
            action.get('scroll_dx', 0),
            action.get('scroll_dy', 0),
            len(self.action_history) / 100
        ]
        return np.array(embedding)
    
    def prepare_input(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare model input from current state"""
        # Capture screen
        screen = self.capture_screen()
        screen_tensor = torch.FloatTensor(screen).permute(2, 0, 1).unsqueeze(0)  # Add batch dim
        
        # Prepare action sequence
        if len(self.action_history) == 0:
            # Initialize with dummy action if no history
            dummy_action = {
                'timestamp': 0, 'time_delta': 0, 'x_norm': 0, 'y_norm': 0,
                'type': 'mouse', 'action': 'move', 'scroll_dx': 0, 'scroll_dy': 0
            }
            action_embeddings = [self.create_action_embedding(dummy_action)]
        else:
            action_embeddings = [self.create_action_embedding(action) 
                               for action in self.action_history[-100:]]  # Last 100 actions
        
        # Pad or truncate to fixed length (100)
        max_length = 100
        if len(action_embeddings) > max_length:
            action_embeddings = action_embeddings[-max_length:]
        else:
            # Pad with zeros at the beginning
            padding_length = max_length - len(action_embeddings)
            padding = [np.zeros(13) for _ in range(padding_length)]
            action_embeddings = padding + action_embeddings
        
        action_tensor = torch.FloatTensor(action_embeddings).unsqueeze(0)  # Add batch dim
        
        return screen_tensor.to(self.device), action_tensor.to(self.device)
    
    def predict_next_action(self) -> Tuple[str, Dict]:
        """Predict next action based on current state"""
        with torch.no_grad():
            screen_tensor, action_tensor = self.prepare_input()
            
            # Get model prediction
            task_logits, next_action_pred, features = self.model(screen_tensor, action_tensor)
            
            # Get task classification
            task_probs = F.softmax(task_logits, dim=1)
            predicted_task_idx = torch.argmax(task_probs, dim=1).item()
            predicted_task = self.label_encoder.inverse_transform([predicted_task_idx])[0]
            confidence = task_probs[0, predicted_task_idx].item()
            
            # Decode next action prediction
            next_action_np = next_action_pred[0].cpu().numpy()
            
            # Interpret action prediction
            action_info = {
                'predicted_task': predicted_task,
                'confidence': confidence,
                'timestamp': next_action_np[0] * 10000,
                'time_delta': next_action_np[1] * 1000,
                'x_norm': next_action_np[2],
                'y_norm': next_action_np[3],
                'is_mouse': next_action_np[4] > 0.5,
                'is_keyboard': next_action_np[5] > 0.5,
                'is_click': next_action_np[6] > 0.5,
                'is_press': next_action_np[7] > 0.5,
                'is_release': next_action_np[8] > 0.5,
                'is_scroll': next_action_np[9] > 0.5,
                'scroll_dx': next_action_np[10],
                'scroll_dy': next_action_np[11]
            }
            
            return predicted_task, action_info
    
    def execute_action(self, action_info: Dict):
        """Execute the predicted action"""
        try:
            # Convert normalized coordinates to screen coordinates
            x = int(action_info['x_norm'] * self.screen_width)
            y = int(action_info['y_norm'] * self.screen_height)
            
            # Ensure coordinates are within screen bounds
            x = max(0, min(x, self.screen_width - 1))
            y = max(0, min(y, self.screen_height - 1))
            
            # Wait based on predicted time delta (but cap it)
            time_delta = min(action_info.get('time_delta', 100), 2000) / 1000.0  # Max 2 seconds
            if time_delta > 0.05:  # Minimum 50ms delay
                time.sleep(time_delta)
            
            # Execute action based on type
            if action_info['is_mouse']:
                if action_info['is_click']:
                    self.logger.info(f"Clicking at ({x}, {y})")
                    pyautogui.click(x, y)
                    
                elif action_info['is_scroll']:
                    scroll_x = int(action_info['scroll_dx'])
                    scroll_y = int(action_info['scroll_dy'])
                    self.logger.info(f"Scrolling ({scroll_x}, {scroll_y}) at ({x}, {y})")
                    pyautogui.scroll(scroll_y, x=x, y=y)
                    
                else:
                    # Move mouse
                    self.logger.info(f"Moving mouse to ({x}, {y})")
                    pyautogui.moveTo(x, y, duration=0.1)
            
            elif action_info['is_keyboard']:
                # For keyboard actions, we'd need more sophisticated key prediction
                # This is a simplified version
                if action_info['is_press']:
                    self.logger.info("Keyboard action detected (simplified)")
                    # Could extend this to predict specific keys
            
            # Record this action in history
            executed_action = {
                'timestamp': time.time() * 1000,
                'time_delta': action_info.get('time_delta', 0),
                'type': 'mouse' if action_info['is_mouse'] else 'keyboard',
                'action': self._determine_action_type(action_info),
                'x_norm': action_info['x_norm'],
                'y_norm': action_info['y_norm'],
                'x_raw': x,
                'y_raw': y,
                'scroll_dx': action_info.get('scroll_dx', 0),
                'scroll_dy': action_info.get('scroll_dy', 0)
            }
            
            self.action_history.append(executed_action)
            
            # Keep history size manageable
            if len(self.action_history) > self.max_history_length:
                self.action_history = self.action_history[-self.max_history_length:]
                
        except Exception as e:
            self.logger.error(f"Error executing action: {e}")
    
    def _determine_action_type(self, action_info: Dict) -> str:
        """Determine the specific action type from predictions"""
        if action_info['is_click']:
            return 'click'
        elif action_info['is_scroll']:
            return 'scroll'
        elif action_info['is_press']:
            return 'press'
        elif action_info['is_release']:
            return 'release'
        else:
            return 'move'
    
    def execute_task(self, target_task: str, max_steps: int = 50, confidence_threshold: float = 0.3):
        """Execute a complete task"""
        self.logger.info(f"Starting task execution: {target_task}")
        self.is_executing = True
        
        steps = 0
        consecutive_low_confidence = 0
        
        try:
            while self.is_executing and steps < max_steps:
                # Predict next action
                predicted_task, action_info = self.predict_next_action()
                
                self.logger.info(f"Step {steps + 1}: Predicted task '{predicted_task}' "
                               f"(confidence: {action_info['confidence']:.3f})")
                
                # Check if we're still on the right task
                if predicted_task != target_task:
                    self.logger.warning(f"Task mismatch: expected '{target_task}', "
                                      f"got '{predicted_task}'")
                
                # Check confidence
                if action_info['confidence'] < confidence_threshold:
                    consecutive_low_confidence += 1
                    self.logger.warning(f"Low confidence: {action_info['confidence']:.3f}")
                    
                    if consecutive_low_confidence >= 3:
                        self.logger.error("Too many low confidence predictions. Stopping.")
                        break
                else:
                    consecutive_low_confidence = 0
                
                # Execute the action
                self.execute_action(action_info)
                
                steps += 1
                
                # Small delay between actions
                time.sleep(0.2)
                
        except KeyboardInterrupt:
            self.logger.info("Task execution interrupted by user")
        except Exception as e:
            self.logger.error(f"Error during task execution: {e}")
        finally:
            self.is_executing = False
            self.logger.info(f"Task execution completed after {steps} steps")
    
    def start_recording_mode(self):
        """Start recording user actions for training data collection"""
        self.logger.info("Starting recording mode...")
        self.recording_data = {
            'start_time': time.time(),
            'actions': [],
            'screenshots': []
        }
        
        def on_click(x, y, button, pressed):
            if pressed:
                timestamp = (time.time() - self.recording_data['start_time']) * 1000
                action = {
                    'timestamp_ms': timestamp,
                    'type': 'mouse',
                    'action': 'click',
                    'button': str(button).split('.')[-1],
                    'coordinates': {'x': x, 'y': y},
                    'screen_size': {'width': self.screen_width, 'height': self.screen_height}
                }
                self.recording_data['actions'].append(action)
                self.logger.info(f"Recorded click at ({x}, {y})")
        
        def on_key(key, pressed):
            timestamp = (time.time() - self.recording_data['start_time']) * 1000
            try:
                key_char = key.char
            except AttributeError:
                key_char = str(key).split('.')[-1]
            
            action = {
                'timestamp_ms': timestamp,
                'type': 'keyboard',
                'action': 'press' if pressed else 'release',
                'key': key_char
            }
            self.recording_data['actions'].append(action)
            
            if pressed:
                self.logger.info(f"Recorded key press: {key_char}")
        
        # Set up listeners
        mouse_listener = mouse.Listener(on_click=on_click)
        keyboard_listener = keyboard.Listener(on_press=lambda key: on_key(key, True),
                                           on_release=lambda key: on_key(key, False))
        
        mouse_listener.start()
        keyboard_listener.start()
        
        self.logger.info("Recording started. Press Ctrl+C to stop.")
        
        try:
            # Periodically capture screenshots
            while True:
                screenshot = ImageGrab.grab()
                screenshot_b64 = self._image_to_base64(screenshot)
                
                screenshot_data = {
                    'timestamp_ms': (time.time() - self.recording_data['start_time']) * 1000,
                    'image_base64': screenshot_b64,
                    'size': {'width': self.screen_width, 'height': self.screen_height}
                }
                self.recording_data['screenshots'].append(screenshot_data)
                
                time.sleep(5)  # Capture screenshot every 5 seconds
                
        except KeyboardInterrupt:
            self.logger.info("Recording stopped")
            
        mouse_listener.stop()
        keyboard_listener.stop()
        
        # Save recorded data
        self.save_recording()
    
    def _image_to_base64(self, image):
        """Convert PIL image to base64 string"""
        import io
        import base64
        
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str
    
    def save_recording(self, filename: str = None):
        """Save recorded data to file"""
        if not hasattr(self, 'recording_data'):
            self.logger.error("No recording data to save")
            return
        
        if filename is None:
            filename = f"recorded_task_{int(time.time())}.json"
        
        # Prepare final data structure
        recording = {
            'task_label': input("Enter task label: ").strip(),
            'start_time': self.recording_data['start_time'],
            'end_time': time.time(),
            'duration_ms': (time.time() - self.recording_data['start_time']) * 1000,
            'screen_size': {'width': self.screen_width, 'height': self.screen_height},
            'screenshots': self.recording_data['screenshots'],
            'actions': self.recording_data['actions']
        }
        
        with open(filename, 'w') as f:
            json.dump(recording, f, indent=2)
        
        self.logger.info(f"Recording saved to {filename}")
    
    def interactive_mode(self):
        """Interactive mode for testing the agent"""
        print("=== GUI AI Agent Interactive Mode ===")
        print("Commands:")
        print("  execute <task_name> - Execute a task")
        print("  record - Start recording mode")
        print("  predict - Show next action prediction")
        print("  history - Show action history")
        print("  quit - Exit")
        
        while True:
            try:
                command = input("\n> ").strip().lower()
                
                if command.startswith('execute '):
                    task_name = command[8:].strip()
                    self.execute_task(task_name)
                    
                elif command == 'record':
                    self.start_recording_mode()
                    
                elif command == 'predict':
                    task, action_info = self.predict_next_action()
                    print(f"Predicted task: {task}")
                    print(f"Confidence: {action_info['confidence']:.3f}")
                    print(f"Next action: {action_info}")
                    
                elif command == 'history':
                    print(f"Action history ({len(self.action_history)} actions):")
                    for i, action in enumerate(self.action_history[-10:]):  # Show last 10
                        print(f"  {i}: {action['type']} {action['action']} at "
                              f"({action['x_raw']}, {action['y_raw']})")
                
                elif command == 'quit':
                    break
                    
                else:
                    print("Unknown command")
                    
            except KeyboardInterrupt:
                print("\nUse 'quit' to exit")
            except Exception as e:
                print(f"Error: {e}")

def main():
    """Main function to run the GUI agent"""
    import argparse
    
    parser = argparse.ArgumentParser(description='GUI AI Agent')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--encoder', required=True, help='Path to label encoder')
    parser.add_argument('--task', help='Task to execute')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('--record', action='store_true', help='Start recording mode')
    
    args = parser.parse_args()
    
    # Initialize agent
    agent = GUIAgent(args.model, args.encoder)
    
    if args.record:
        agent.start_recording_mode()
    elif args.interactive:
        agent.interactive_mode()
    elif args.task:
        agent.execute_task(args.task)
    else:
        print("No action specified. Use --help for options.")

if __name__ == "__main__":
    main()

# python execution_agent.py --model final_gui_model.pth --encoder label_encoder.pkl --interactive