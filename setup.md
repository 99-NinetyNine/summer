# UI Action Automation System

🤖 **An AI system that learns from your screen interactions and automates UI tasks**

This system can observe screenshots and actions (like your example data), learn patterns, and then execute similar tasks automatically. Perfect for automating repetitive UI workflows like logins, form filling, and web navigation.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Setup Environment
```bash
python complete_training_system.py setup
```

This creates the directory structure:
```
ui_action_automation/
├── data/                   # Put your JSON data files here
├── models/                 # Trained models will be saved here
├── experiments/           # Training experiments and results
├── enhanced_data/         # Augmented training data
├── configs/              # Configuration files
├── logs/                 # Training logs
└── plots/                # Evaluation plots
```

### 3. Prepare Your Data

Put your JSON files (like the one you showed me) in the `data/` folder. Your data format is perfect:

```json
{
  "task_label": "login",
  "screenshots": [...],
  "actions": [
    {
      "timestamp_ms": 1443,
      "type": "mouse", 
      "action": "click",
      "coordinates": {"x": 870, "y": 547}
    },
    {
      "timestamp_ms": 2067,
      "type": "keyboard",
      "action": "press", 
      "key": "d"
    }
  ]
}
```

### 4. Train Your Model

**Simple training:**
```bash
python complete_training_system.py --data-dir data/ --experiment-name my_login_bot
```

**Advanced training with config:**
```bash
python complete_training_system.py --config configs/default_config.json
```

### 5. Run Automation

```python
from ui_action_system import RealTimeExecutionAgent

# Initialize with your trained model
agent = RealTimeExecutionAgent(
    model_path='experiments/my_login_bot/models/best_model.pth',
    action_encoder_path='experiments/my_login_bot/models/action_encoder.pkl'
)

# Run automation
agent.run_automation('login', max_actions=20, confidence_threshold=0.7)
```

## 📋 File Structure

| File | Purpose |
|------|---------|
| `ui_action_system.py` | Core model, dataset, and execution agent |
| `data_augmentation.py` | Advanced data augmentation (8x+ dataset expansion) |
| `complete_training_system.py` | Full training pipeline with evaluation |
| `example_usage.py` | Example scripts showing how to use everything |
| `requirements.txt` | Python dependencies |

## 🔧 How It Works

1. **Data Processing**: Loads your JSON files with screenshots and actions
2. **Data Augmentation**: Creates variations (different brightness, small coordinate shifts, timing changes)
3. **Model Training**: CNN learns to predict actions from screenshots
4. **Real-time Execution**: Captures screen, predicts action, executes it

### Model Architecture
```
Screenshot (1920x1080) → Resize → CNN Feature Extractor → Two Heads:
                                                        ├─ Action Classification
                                                        └─ Coordinate Regression
```

## 🎯 Examples

### Example 1: Simple Training
```python
from ui_action_system import train_model

data_files = ["data/login_task.json"]
train_model(data_files, "my_model.pth", "my_encoder.pkl")
```

### Example 2: Data Augmentation
```python
from data_augmentation import DatasetEnhancer

enhancer = DatasetEnhancer("augmented_data")
enhanced_files = enhancer.enhance_dataset(
    ["data/login_task.json"], 
    enhancement_factor=8  # Creates 8x more training data
)
```

### Example 3: Real-time Execution
```python
from ui_action_system import RealTimeExecutionAgent

agent = RealTimeExecutionAgent("model.pth", "encoder.pkl")

# Just predict (don't execute)
screenshot = agent.capture_screen()
action = agent.predict_action(screenshot)
print(f"Would do: {action}")

# Actually execute
agent.run_automation("login_task")
```

### Example 4: Run All Examples
```bash
python example_usage.py
```

## ⚙️ Configuration

Edit `configs/default_config.json`:

```json
{
  "experiment_name": "my_automation_bot",
  "data_files": ["data/login_task_1.json", "data/login_task_2.json"],
  "batch_size": 8,
  "num_epochs": 50,
  "enhance_data": true,
  "enhancement_factor": 8
}
```

## 📊 Training Features

- **Smart Data Augmentation**: Preserves click accuracy while adding visual variety
- **Multi-task Learning**: Train on different UI tasks simultaneously  
- **Early Stopping**: Prevents overfitting
- **Comprehensive Evaluation**: Confusion matrices, pixel-level accuracy
- **Experiment Tracking**: Detailed logs and metrics

## 🎮 Real-time Features

- **Confidence Thresholds**: Only execute high-confidence predictions
- **Safety Limits**: Maximum action counts, failsafe mechanisms
- **Live Monitoring**: See what the AI is thinking in real-time

## 🔄 Data Augmentation Magic

Your single login recording becomes:
- ✅ 8 visual variations (brightness, contrast, etc.)
- ✅ Different screen sizes and zoom levels
- ✅ Browser variations
- ✅ Small coordinate variations (±5 pixels)
- ✅ Timing variations
- ✅ Task variations (login_gmail, login_facebook, etc.)

**Result**: 1 recording → 50+ training examples!

## 🎯 Perfect for Your Use Case

Since you collected data with pyautogui, this system:
- ✅ Uses your exact data format
- ✅ Handles mouse clicks and keyboard input
- ✅ Preserves coordinate accuracy
- ✅ Supports multiple login sites
- ✅ Executes with pyautogui (same as your collection)

## 🚨 Safety Features

- `pyautogui.FAILSAFE = True` (move mouse to corner to stop)
- Confidence thresholds (only execute if AI is confident)
- Maximum action limits
- Detailed logging of all actions

## 🐛 Troubleshooting

### "No data found"
- Check your JSON files are in `data/` folder
- Run validation: `validate_data_format('data/your_file.json')`

### "Low accuracy"
- Increase `enhancement_factor` for more training data
- Add more original recordings
- Check data quality

### "Execution errors"
- Verify screen resolution matches training data
- Adjust confidence thresholds
- Check pyautogui permissions

## 📈 Advanced Usage

### Custom Action Types
Extend the system for new UI patterns:
```python
# Add custom action handling in ui_action_system.py
def handle_custom_action(self, action):
    if action['type'] == 'drag':
        # Your drag logic
        pass
```

### Multi-Monitor Support
```python
# Capture specific monitor
screenshot = pyautogui.screenshot(region=(0, 0, 1920, 1080))
```

### Integration with Other Tools
```python
# Use with Selenium, Playwright, etc.
from selenium import webdriver
driver = webdriver.Chrome()
# Your automation logic
```

## 🎉 What Makes This Special

1. **Works with Your Data**: Uses your pyautogui-collected JSON format exactly
2. **Smart Augmentation**: Expands dataset while preserving accuracy
3. **Production Ready**: Real error handling, logging, safety features
4. **Multi-Task**: One model can handle login, search, forms, etc.
5. **Easy to Use**: Simple Python scripts to get started

## 🔮 Next Steps

1. **Start Simple**: Use `example_usage.py` to see it working
2. **Collect More Data**: Record different websites/scenarios
3. **Train & Test**: Start with small datasets, then scale up
4. **Deploy**: Use the execution agent for real automation

---

**Ready to automate your UI tasks?** 

```bash
# Get started now!
python complete_training_system.py setup
python example_usage.py
```

Your login automation journey begins here! 🚀