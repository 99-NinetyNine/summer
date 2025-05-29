# data_augmentation.py
"""
Advanced Data Augmentation System for UI Interaction Data
Expands your dataset with realistic variations while preserving action accuracy
"""

import json
import base64
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io
import random
from typing import List, Dict, Tuple, Optional
import albumentations as A
from pathlib import Path
import copy
import math
import logging

logger = logging.getLogger(__name__)

class AdvancedUIDataAugmenter:
    """Advanced augmentation specifically for UI interaction data"""
    
    def __init__(self, preserve_ui_elements: bool = True):
        self.preserve_ui_elements = preserve_ui_elements
        self.setup_augmentations()
    
    def setup_augmentations(self):
        """Setup different types of augmentations"""
        
        # Visual augmentations (safe for UI)
        self.visual_transforms = A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=0.15, 
                contrast_limit=0.15, 
                p=0.4
            ),
            A.HueSaturationValue(
                hue_shift_limit=8, 
                sat_shift_limit=15, 
                val_shift_limit=15, 
                p=0.3
            ),
            A.GaussNoise(var_limit=(5.0, 25.0), p=0.2),
            A.RandomGamma(gamma_limit=(85, 115), p=0.3),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=0.2),
        ])
        
        # Geometric augmentations (minimal to preserve click coordinates)
        self.geometric_transforms = A.Compose([
            A.ShiftScaleRotate(
                shift_limit=0.02,  # Very small shifts
                scale_limit=0.03,  # Very small scaling
                rotate_limit=1,    # Minimal rotation
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.3
            ),
        ])
        
        # Screen simulation augmentations
        self.screen_simulation = [
            self._simulate_different_screen_sizes,
            self._simulate_different_browsers,
            self._simulate_zoom_levels,
            self._add_cursor_variations,
        ]
    
    def augment_sequence(self, sequence_data: Dict, num_augmentations: int = 5) -> List[Dict]:
        """Generate multiple augmented versions of a sequence"""
        augmented_sequences = []
        
        for i in range(num_augmentations):
            # Create a copy of the original sequence
            aug_sequence = copy.deepcopy(sequence_data)
            
            # Apply random augmentations
            aug_sequence = self._apply_random_augmentations(aug_sequence)
            
            # Add variation identifier
            aug_sequence['augmentation_id'] = i
            aug_sequence['original_task'] = sequence_data.get('task_label', 'unknown')
            aug_sequence['task_label'] = f"{sequence_data.get('task_label', 'unknown')}_aug_{i}"
            
            augmented_sequences.append(aug_sequence)
        
        return augmented_sequences
    
    def _apply_random_augmentations(self, sequence: Dict) -> Dict:
        """Apply random augmentations to a sequence"""
        
        # Augment screenshots
        if 'screenshots' in sequence:
            for screenshot in sequence['screenshots']:
                screenshot['image_base64'] = self._augment_screenshot(
                    screenshot['image_base64']
                )
        
        # Apply action variations
        if 'actions' in sequence:
            sequence['actions'] = self._augment_actions(sequence['actions'])
        
        # Apply timing variations
        sequence = self._add_timing_variations(sequence)
        
        return sequence
    
    def _augment_screenshot(self, base64_image: str) -> str:
        """Augment a single screenshot"""
        try:
            # Decode image
            img_data = base64.b64decode(base64_image)
            img = Image.open(io.BytesIO(img_data))
            img_array = np.array(img)
            
            # Apply visual augmentations
            if random.random() < 0.7:
                augmented = self.visual_transforms(image=img_array)
                img_array = augmented['image']
            
            # Apply geometric augmentations (carefully)
            if random.random() < 0.3:
                augmented = self.geometric_transforms(image=img_array)
                img_array = augmented['image']
            
            # Apply screen simulation
            if random.random() < 0.4:
                aug_func = random.choice(self.screen_simulation)
                img_array = aug_func(img_array)
            
            # Convert back to base64
            img_pil = Image.fromarray(img_array)
            buffer = io.BytesIO()
            img_pil.save(buffer, format='PNG')
            augmented_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            return augmented_base64
            
        except Exception as e:
            logger.warning(f"Screenshot augmentation failed: {e}")
            return base64_image  # Return original if augmentation fails
    
    def _augment_actions(self, actions: List[Dict]) -> List[Dict]:
        """Add variations to actions"""
        augmented_actions = []
        
        for action in actions:
            aug_action = copy.deepcopy(action)
            
            # Add small coordinate variations for mouse actions
            if action.get('type') == 'mouse' and 'coordinates' in action:
                coords = action['coordinates']
                if coords:
                    # Add small random offset (±5 pixels)
                    offset_x = random.randint(-5, 5)
                    offset_y = random.randint(-5, 5)
                    
                    aug_action['coordinates'] = {
                        'x': max(0, coords['x'] + offset_x),
                        'y': max(0, coords['y'] + offset_y)
                    }
            
            # Add timing variations
            if random.random() < 0.3:
                time_offset = random.randint(-100, 100)  # ±100ms
                aug_action['timestamp_ms'] = max(0, action['timestamp_ms'] + time_offset)
            
            augmented_actions.append(aug_action)
        
        return augmented_actions
    
    def _add_timing_variations(self, sequence: Dict) -> Dict:
        """Add realistic timing variations"""
        if 'duration_ms' in sequence:
            # Add ±10% variation to total duration
            variation = int(sequence['duration_ms'] * 0.1)
            offset = random.randint(-variation, variation)
            sequence['duration_ms'] = max(1000, sequence['duration_ms'] + offset)
        
        return sequence
    
    def _simulate_different_screen_sizes(self, image: np.ndarray) -> np.ndarray:
        """Simulate different screen sizes"""
        original_size = image.shape[:2]
        
        # Common screen sizes
        screen_sizes = [
            (1366, 768),   # HD
            (1920, 1080),  # Full HD
            (2560, 1440),  # QHD
            (1440, 900),   # MacBook
            (1280, 720),   # HD Ready
        ]
        
        target_size = random.choice(screen_sizes)
        
        # Resize and pad/crop as needed
        resized = cv2.resize(image, target_size)
        
        # If original was larger, we might need to adjust back
        if original_size != target_size:
            resized = cv2.resize(resized, (original_size[1], original_size[0]))
        
        return resized
    
    def _simulate_different_browsers(self, image: np.ndarray) -> np.ndarray:
        """Simulate different browser interfaces"""
        # Add browser-specific UI elements (simplified)
        h, w = image.shape[:2]
        
        # Simulate different browser header heights
        header_heights = [60, 80, 100, 120]
        header_height = random.choice(header_heights)
        
        # Add colored header
        header_colors = [
            [240, 240, 240],  # Light gray (Chrome-like)
            [230, 230, 230],  # Slightly darker gray
            [250, 250, 250],  # Very light gray
            [220, 220, 220],  # Medium gray
        ]
        
        header_color = random.choice(header_colors)
        
        # Create modified image with browser header simulation
        modified_image = image.copy()
        if random.random() < 0.3:  # Apply browser simulation occasionally
            # Shift content down to simulate browser header
            shifted_content = np.zeros_like(modified_image)
            shifted_content[header_height:, :] = modified_image[:-header_height, :]
            shifted_content[:header_height, :] = header_color
            modified_image = shifted_content
        
        return modified_image
    
    def _simulate_zoom_levels(self, image: np.ndarray) -> np.ndarray:
        """Simulate different browser zoom levels"""
        zoom_factors = [0.9, 0.95, 1.0, 1.05, 1.1, 1.25]
        zoom = random.choice(zoom_factors)
        
        if zoom != 1.0:
            h, w = image.shape[:2]
            new_h, new_w = int(h * zoom), int(w * zoom)
            
            # Resize image
            resized = cv2.resize(image, (new_w, new_h))
            
            # Crop or pad to original size
            if zoom > 1.0:  # Crop center
                start_y = (new_h - h) // 2
                start_x = (new_w - w) // 2
                cropped = resized[start_y:start_y+h, start_x:start_x+w]
                return cropped
            else:  # Pad with background color
                padded = np.full((h, w, 3), fill_value=240, dtype=np.uint8)
                start_y = (h - new_h) // 2
                start_x = (w - new_w) // 2
                padded[start_y:start_y+new_h, start_x:start_x+new_w] = resized
                return padded
        
        return image
    
    def _add_cursor_variations(self, image: np.ndarray) -> np.ndarray:
        """Add different cursor styles (simulated)"""
        # This is a simplified version - in reality, cursors aren't part of screenshots
        # But we can simulate their presence for training variety
        if random.random() < 0.2:  # Rarely add cursor simulation
            h, w = image.shape[:2]
            
            # Random cursor position
            cursor_x = random.randint(50, w - 50)
            cursor_y = random.randint(50, h - 50)
            
            # Add small cursor-like mark (just a few pixels)
            cursor_color = [0, 0, 0]  # Black cursor
            image[cursor_y:cursor_y+2, cursor_x:cursor_x+2] = cursor_color
        
        return image

class TaskVariationGenerator:
    """Generate variations of tasks to increase dataset diversity"""
    
    def __init__(self):
        self.task_templates = {
            'login': {
                'variations': [
                    'login_gmail', 'login_facebook', 'login_twitter', 
                    'login_github', 'login_linkedin', 'login_microsoft'
                ],
                'common_elements': ['email_field', 'password_field', 'login_button']
            },
            'search': {
                'variations': [
                    'search_google', 'search_youtube', 'search_amazon',
                    'search_stackoverflow', 'search_github'
                ],
                'common_elements': ['search_box', 'search_button', 'results']
            },
            'form_fill': {
                'variations': [
                    'contact_form', 'registration_form', 'survey_form',
                    'checkout_form', 'profile_form'
                ],
                'common_elements': ['text_fields', 'dropdown', 'submit_button']
            }
        }
    
    def generate_task_variations(self, original_sequence: Dict) -> List[Dict]:
        """Generate task variations based on the original task"""
        base_task = original_sequence.get('task_label', 'unknown')
        variations = []
        
        # Find matching task template
        task_type = self._identify_task_type(base_task)
        
        if task_type in self.task_templates:
            template = self.task_templates[task_type]
            
            for variation_name in template['variations']:
                if variation_name != base_task:  # Don't duplicate original
                    varied_sequence = copy.deepcopy(original_sequence)
                    varied_sequence['task_label'] = variation_name
                    varied_sequence['task_variation'] = True
                    varied_sequence['base_task'] = base_task
                    variations.append(varied_sequence)
        
        return variations
    
    def _identify_task_type(self, task_label: str) -> str:
        """Identify the general type of task"""
        task_lower = task_label.lower()
        
        if 'login' in task_lower or 'signin' in task_lower:
            return 'login'
        elif 'search' in task_lower:
            return 'search'
        elif 'form' in task_lower or 'fill' in task_lower:
            return 'form_fill'
        
        return 'unknown'

class SequentialActionGenerator:
    """Generate realistic sequential actions for training"""
    
    def __init__(self):
        self.action_sequences = {
            'typing': self._generate_typing_sequence,
            'navigation': self._generate_navigation_sequence,
            'form_interaction': self._generate_form_sequence,
        }
    
    def generate_intermediate_actions(self, sequence: Dict) -> Dict:
        """Generate intermediate actions to make sequences more realistic"""
        actions = sequence.get('actions', [])
        if len(actions) < 2:
            return sequence
        
        enhanced_actions = []
        
        for i, action in enumerate(actions):
            enhanced_actions.append(action)
            
            # Add intermediate actions between major actions
            if i < len(actions) - 1:
                next_action = actions[i + 1]
                intermediate = self._generate_intermediate_action(action, next_action)
                if intermediate:
                    enhanced_actions.extend(intermediate)
        
        sequence['actions'] = enhanced_actions
        return sequence
    
    def _generate_intermediate_action(self, current: Dict, next_action: Dict) -> List[Dict]:
        """Generate realistic intermediate actions"""
        intermediate_actions = []
        
        # Add mouse movement before clicks
        if (next_action.get('type') == 'mouse' and 
            next_action.get('action') == 'click' and 
            current.get('type') != 'mouse'):
            
            mouse_move = {
                'timestamp_ms': current['timestamp_ms'] + 50,
                'type': 'mouse',
                'action': 'move',
                'coordinates': next_action.get('coordinates', {'x': 0, 'y': 0})
            }
            intermediate_actions.append(mouse_move)
        
        # Add pauses between rapid actions
        time_diff = next_action['timestamp_ms'] - current['timestamp_ms']
        if time_diff < 100:  # Very fast actions
            pause_action = {
                'timestamp_ms': current['timestamp_ms'] + time_diff // 2,
                'type': 'system',
                'action': 'pause',
                'duration': 50
            }
            intermediate_actions.append(pause_action)
        
        return intermediate_actions
    
    def _generate_typing_sequence(self, text: str, start_time: int) -> List[Dict]:
        """Generate realistic typing sequence"""
        actions = []
        current_time = start_time
        
        for char in text:
            # Random typing speed (50-200ms per character)
            char_delay = random.randint(50, 200)
            
            # Key press
            actions.append({
                'timestamp_ms': current_time,
                'type': 'keyboard',
                'action': 'press',
                'key': char,
                'key_code': ord(char)
            })
            
            # Key release
            actions.append({
                'timestamp_ms': current_time + random.randint(20, 80),
                'type': 'keyboard',
                'action': 'release',
                'key': char
            })
            
            current_time += char_delay
        
        return actions
    
    def _generate_navigation_sequence(self, url: str, start_time: int) -> List[Dict]:
        """Generate browser navigation sequence"""
        # This is a placeholder for navigation action generation
        return []
    
    def _generate_form_sequence(self, form_data: Dict, start_time: int) -> List[Dict]:
        """Generate form filling sequence"""
        # This is a placeholder for form action generation
        return []

class DatasetEnhancer:
    """Main class to enhance and expand the dataset"""
    
    def __init__(self, output_dir: str = "enhanced_dataset"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.augmenter = AdvancedUIDataAugmenter()
        self.task_generator = TaskVariationGenerator()
        self.action_generator = SequentialActionGenerator()
    
    def enhance_dataset(self, input_files: List[str], enhancement_factor: int = 5) -> List[str]:
        """Enhance dataset with multiple augmentation techniques"""
        enhanced_files = []
        
        for input_file in input_files:
            logger.info(f"Processing {input_file}...")
            
            try:
                # Load original data
                with open(input_file, 'r') as f:
                    original_data = json.load(f)
                
                enhanced_sequences = []
                
                # 1. Visual and timing augmentations
                visual_augmented = self.augmenter.augment_sequence(
                    original_data, num_augmentations=enhancement_factor
                )
                enhanced_sequences.extend(visual_augmented)
                
                # 2. Task variations
                task_variations = self.task_generator.generate_task_variations(original_data)
                enhanced_sequences.extend(task_variations)
                
                # 3. Action sequence enhancements
                for seq in enhanced_sequences[:]:  # Copy list to avoid modifying during iteration
                    enhanced_seq = self.action_generator.generate_intermediate_actions(seq)
                    enhanced_sequences.append(enhanced_seq)
                
                # Save enhanced sequences
                for i, enhanced_seq in enumerate(enhanced_sequences):
                    output_filename = f"{Path(input_file).stem}_enhanced_{i}.json"
                    output_path = self.output_dir / output_filename
                    
                    with open(output_path, 'w') as f:
                        json.dump(enhanced_seq, f, indent=2)
                    
                    enhanced_files.append(str(output_path))
                    
            except Exception as e:
                logger.error(f"Failed to process {input_file}: {e}")
        
        logger.info(f"Generated {len(enhanced_files)} enhanced data files")
        return enhanced_files
    
    def create_balanced_dataset(self, input_files: List[str]) -> str:
        """Create a balanced dataset with equal representation"""
        all_sequences = []
        task_counts = {}
        
        # Load all sequences and count tasks
        for file_path in input_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    task_label = data.get('task_label', 'unknown')
                    
                    if task_label not in task_counts:
                        task_counts[task_label] = []
                    
                    task_counts[task_label].append(data)
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
        
        # Balance dataset
        if task_counts:
            max_samples = max(len(sequences) for sequences in task_counts.values())
            balanced_sequences = []
            
            for task_label, sequences in task_counts.items():
                # Upsample if needed
                while len(sequences) < max_samples:
                    # Randomly select and augment existing sequences
                    base_seq = random.choice(sequences)
                    augmented = self.augmenter.augment_sequence(base_seq, num_augmentations=1)[0]
                    sequences.append(augmented)
                
                balanced_sequences.extend(sequences[:max_samples])
            
            # Save balanced dataset
            balanced_file = self.output_dir / "balanced_dataset.json"
            with open(balanced_file, 'w') as f:
                json.dump(balanced_sequences, f, indent=2)
            
            logger.info(f"Created balanced dataset with {len(balanced_sequences)} sequences")
            logger.info(f"Task distribution: {dict((k, len(v)) for k, v in task_counts.items())}")
            
            return str(balanced_file)
        else:
            logger.error("No valid sequences found for balancing")
            return ""

# Usage example
def main():
    """Example usage of the data augmentation system"""
    # Initialize dataset enhancer
    enhancer = DatasetEnhancer("enhanced_ui_dataset")
    
    # Input data files
    input_files = [
        "/home/s/Desktop/ai/data/episode_login_to_website_20250529_095505.json",
        # Add more data files here
    ]
    
    # Enhance dataset
    enhanced_files = enhancer.enhance_dataset(input_files, enhancement_factor=8)
    
    # Create balanced dataset
    balanced_dataset = enhancer.create_balanced_dataset(enhanced_files)
    
    print(f"Dataset enhancement complete!")
    print(f"Enhanced files: {len(enhanced_files)}")
    print(f"Balanced dataset: {balanced_dataset}")

if __name__ == "__main__":
    main()