# complete_integration.py
"""
Complete Integration: From Data Collection to Reward-Based Learning
This shows the full pipeline from your JSON data to a smart, self-improving UI automation system
"""

import json
import time
from pathlib import Path
import logging

# Import all our system components
from ui_action_system import train_model, RealTimeExecutionAgent
from complete_training_system import AdvancedTrainingPipeline, create_config_template
from reward_learning_system import RewardBasedExecutionAgent
from reward_config_system import (
    RewardBasedTrainingOrchestrator, TaskSpecificRewards, 
    train_login_automation, RewardConfig
)

logger = logging.getLogger(__name__)

class UIAutomationMasterPipeline:
    """Master pipeline that handles everything from data to deployment"""
    
    def __init__(self, project_name: str = "ui_automation_project"):
        self.project_name = project_name
        self.project_dir = Path(project_name)
        self.setup_project_structure()
        
        # Track all models and results
        self.models = {}
        self.training_results = {}
        
    def setup_project_structure(self):
        """Setup complete project structure"""
        directories = [
            'data',
            'models/base_models',
            'models/reward_models', 
            'experiments',
            'reward_training',
            'logs',
            'reports',
            'configs'
        ]
        
        for dir_path in directories:
            (self.project_dir / dir_path).mkdir(parents=True, exist_ok=True)
        
        print(f"✅ Project structure created: {self.project_dir}")
    
    def step1_prepare_data(self, data_files: list) -> str:
        """Step 1: Prepare and validate your data"""
        print("\n🔧 Step 1: Preparing Data")
        print("=" * 40)
        
        # Copy data files to project
        valid_files = []
        for file_path in data_files:
            if Path(file_path).exists():
                # Copy to project data directory
                target_path = self.project_dir / 'data' / Path(file_path).name
                
                with open(file_path, 'r') as src, open(target_path, 'w') as dst:
                    dst.write(src.read())
                
                valid_files.append(str(target_path))
                print(f"  ✅ Copied: {file_path} → {target_path}")
            else:
                print(f"  ❌ File not found: {file_path}")
        
        if valid_files:
            print(f"  📊 Total valid data files: {len(valid_files)}")
            return valid_files
        else:
            raise ValueError("No valid data files found!")
    
    def step2_train_base_model(self, data_files: list) -> dict:
        """Step 2: Train base model with data augmentation"""
        print("\n🤖 Step 2: Training Base Model")
        print("=" * 40)
        
        # Configure training
        config = {
            'experiment_name': f'{self.project_name}_base_model',
            'data_files': data_files,
            'output_dir': str(self.project_dir / 'experiments' / 'base_model'),
            'batch_size': 8,
            'num_epochs': 30,
            'enhance_data': True,
            'enhancement_factor': 8,  # 8x more data through augmentation
        }
        
        # Train model
        pipeline = AdvancedTrainingPipeline(config)
        model_path = pipeline.train()
        
        # Store results
        self.models['base_model'] = {
            'model_path': model_path,
            'encoder_path': str(Path(model_path).parent / 'action_encoder.pkl'),
            'config': config
        }
        
        print(f"  ✅ Base model trained: {model_path}")
        return self.models['base_model']
    
    def step3_setup_reward_learning(self, task_type: str = 'login') -> dict:
        """Step 3: Setup reward-based learning"""
        print(f"\n🎯 Step 3: Setting up Reward-Based Learning ({task_type})")
        print("=" * 50)
        
        if 'base_model' not in self.models:
            raise ValueError("Base model must be trained first!")
        
        base_model = self.models['base_model']
        
        # Create reward configuration
        if task_type == 'login':
            reward_config = TaskSpecificRewards.login_task()
            print("  🔑 Using optimized LOGIN reward configuration")
        elif task_type == 'search':
            reward_config = TaskSpecificRewards.search_task()
            print("  🔍 Using optimized SEARCH reward configuration")
        elif task_type == 'form_fill':
            reward_config = TaskSpecificRewards.form_filling_task()
            print("  📝 Using optimized FORM FILLING reward configuration")
        else:
            reward_config = RewardConfig()
            print("  ⚙️ Using default reward configuration")
        
        # Print reward structure
        print(f"  💰 Task completion reward: +{reward_config.task_completion_reward}")
        print(f"  📈 Progress step reward: +{reward_config.progress_step_reward}")
        print(f"  ⚡ Action penalty: {reward_config.action_penalty}")
        print(f"  🎯 Max episodes: {reward_config.max_episodes}")
        
        # Initialize reward-based agent
        agent = RewardBasedExecutionAgent(
            base_model['model_path'],
            base_model['encoder_path'],
            task_type
        )
        
        self.models['reward_agent'] = {
            'agent': agent,
            'config': reward_config,
            'task_type': task_type
        }
        
        print("  ✅ Reward-based learning agent initialized")
        return self.models['reward_agent']
    
    def step4_run_reward_training(self, episodes: int = None) -> str:
        """Step 4: Run reward-based training"""
        print("\n🏃 Step 4: Running Reward-Based Training")
        print("=" * 45)
        
        if 'reward_agent' not in self.models:
            raise ValueError("Reward agent must be setup first!")
        
        reward_info = self.models['reward_agent']
        agent = reward_info['agent']
        config = reward_info['config']
        task_type = reward_info['task_type']
        
        # Use provided episodes or config default
        max_episodes = episodes or config.max_episodes
        
        print(f"  🎮 Starting {max_episodes} episodes of reward-based learning")
        print(f"  📋 Task type: {task_type.upper()}")
        print(f"  ⏱️ Estimated time: {max_episodes * 2} minutes")
        
        # Start reward-based training
        start_time = time.time()
        
        agent.run_reward_based_automation(
            task_name=f"{task_type}_reward_training",
            max_episodes=max_episodes,
            max_actions_per_episode=config.max_actions_per_episode
        )
        
        training_time = time.time() - start_time
        
        # Store training results
        results_dir = agent.save_dir
        self.training_results['reward_training'] = {
            'results_dir': str(results_dir),
            'training_time_minutes': training_time / 60,
            'episodes_completed': max_episodes,
            'task_type': task_type
        }
        
        print(f"  ✅ Reward training completed in {training_time/60:.1f} minutes")
        print(f"  📊 Results saved to: {results_dir}")
        
        return str(results_dir)
    
    def step5_deploy_smart_agent(self) -> 'SmartUIAgent':
        """Step 5: Deploy the trained smart agent"""
        print("\n🚀 Step 5: Deploying Smart UI Agent")
        print("=" * 40)
        
        if 'reward_agent' not in self.models:
            raise ValueError("Reward training must be completed first!")
        
        # Load the best reward-based model
        reward_info = self.models['reward_agent']
        agent = reward_info['agent']
        
        # Create deployment-ready agent
        smart_agent = SmartUIAgent(agent, self.project_name)
        
        print("  🤖 Smart UI Agent deployed and ready!")
        print("  🎯 Agent can now:")
        print("    - Predict actions from screenshots")
        print("    - Execute actions automatically")
        print("    - Learn from success/failure")
        print("    - Adapt to new scenarios")
        
        return smart_agent
    
    def generate_project_report(self) -> str:
        """Generate comprehensive project report"""
        print("\n📊 Generating Project Report")
        print("=" * 35)
        
        report = f"""
# UI Automation Project Report
## Project: {self.project_name}

### Training Summary
"""
        
        if 'base_model' in self.models:
            report += f"""
#### Base Model Training
- ✅ Model: {self.models['base_model']['model_path']}
- ✅ Encoder: {self.models['base_model']['encoder_path']}
- 🔧 Data augmentation: 8x expansion
"""
        
        if 'reward_training' in self.training_results:
            result = self.training_results['reward_training']
            report += f"""
#### Reward-Based Learning
- ✅ Training completed: {result['episodes_completed']} episodes
- ⏱️ Training time: {result['training_time_minutes']:.1f} minutes
- 🎯 Task type: {result['task_type'].upper()}
- 📊 Results: {result['results_dir']}
"""
        
        report += f"""

### Usage Instructions

#### Quick Start
```python
from complete_integration import load_project
agent = load_project('{self.project_name}')

# Run automation
agent.automate_task('login', confidence_threshold=0.8)
```

#### Advanced Usage
```python
# Predict without executing
action = agent.predict_next_action()
print(f"Would execute: {{action}}")

# Execute with confirmation
if action['confidence'] > 0.9:
    agent.execute_action(action)
```

### Project Structure
```
{self.project_name}/
├── data/                   # Training data
├── models/                 # Trained models
│   ├── base_models/       # Initial trained models
│   └── reward_models/     # Reward-optimized models
├── experiments/           # Training experiments
├── reward_training/       # Reward learning results
├── logs/                  # Training logs
├── reports/              # Generated reports
└── configs/              # Configuration files
```

### Next Steps
1. Test the deployed agent on real scenarios
2. Collect more training data for edge cases
3. Fine-tune reward configurations
4. Deploy to production environment

---
*Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        # Save report
        report_path = self.project_dir / 'reports' / 'project_report.md'
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"  📄 Report saved to: {report_path}")
        return str(report_path)

class SmartUIAgent:
    """Production-ready smart UI agent"""
    
    def __init__(self, reward_agent: RewardBasedExecutionAgent, project_name: str):
        self.reward_agent = reward_agent
        self.project_name = project_name
        self.session_stats = {
            'actions_executed': 0,
            'tasks_completed': 0,
            'session_start': time.time()
        }
    
    def predict_next_action(self) -> dict:
        """Predict next action from current screen"""
        screenshot = self.reward_agent.capture_screen()
        prediction = self.reward_agent.predict_action(screenshot)
        
        return {
            'type': prediction['type'],
            'action': prediction['action'],
            'coordinates': prediction['coordinates'],
            'confidence': prediction['confidence'],
            'key': prediction.get('key')
        }
    
    def execute_action(self, action: dict = None):
        """Execute action (predicted or provided)"""
        if action is None:
            action = self.predict_next_action()
        
        if action['confidence'] < 0.5:
            print(f"⚠️ Low confidence action ({action['confidence']:.2f}), skipping")
            return False
        
        self.reward_agent.execute_action(action)
        self.session_stats['actions_executed'] += 1
        
        print(f"✅ Executed: {action['type']} - {action['action']} (confidence: {action['confidence']:.2f})")
        return True
    
    def automate_task(self, task_name: str, max_actions: int = 30, confidence_threshold: float = 0.7):
        """Automate a complete task"""
        print(f"🤖 Starting automation: {task_name}")
        
        self.reward_agent.run_automation(
            task_name, 
            max_actions=max_actions, 
            confidence_threshold=confidence_threshold
        )
        
        self.session_stats['tasks_completed'] += 1
        print(f"✅ Task '{task_name}' automation completed")
    
    def get_session_stats(self) -> dict:
        """Get current session statistics"""
        runtime = time.time() - self.session_stats['session_start']
        return {
            **self.session_stats,
            'session_runtime_minutes': runtime / 60,
            'actions_per_minute': self.session_stats['actions_executed'] / (runtime / 60) if runtime > 0 else 0
        }
    
    def continuous_learning_mode(self, task_name: str, learning_episodes: int = 10):
        """Run in continuous learning mode"""
        print(f"🧠 Starting continuous learning mode for {task_name}")
        print(f"   Will run {learning_episodes} learning episodes")
        
        self.reward_agent.run_reward_based_automation(
            task_name=f"{task_name}_continuous_learning",
            max_episodes=learning_episodes,
            max_actions_per_episode=25
        )
        
        print("🎓 Continuous learning completed - Agent improved!")

# Convenience functions for easy project management
def create_new_project(project_name: str, data_files: list, task_type: str = 'login') -> SmartUIAgent:
    """Create a complete new UI automation project from scratch"""
    print(f"🚀 Creating new UI automation project: {project_name}")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = UIAutomationMasterPipeline(project_name)
    
    # Run complete pipeline
    try:
        # Step 1: Prepare data
        valid_files = pipeline.step1_prepare_data(data_files)
        
        # Step 2: Train base model
        base_model = pipeline.step2_train_base_model(valid_files)
        
        # Step 3: Setup reward learning
        reward_setup = pipeline.step3_setup_reward_learning(task_type)
        
        # Step 4: Run reward training
        results_dir = pipeline.step4_run_reward_training()
        
        # Step 5: Deploy smart agent
        smart_agent = pipeline.step5_deploy_smart_agent()
        
        # Generate report
        report_path = pipeline.generate_project_report()
        
        print(f"\n🎉 PROJECT CREATED SUCCESSFULLY!")
        print(f"📁 Project directory: {pipeline.project_dir}")
        print(f"📊 Report: {report_path}")
        print(f"🤖 Smart agent ready for use!")
        
        return smart_agent
        
    except Exception as e:
        print(f"❌ Project creation failed: {e}")
        raise

def load_project(project_name: str) -> SmartUIAgent:
    """Load an existing UI automation project"""
    project_dir = Path(project_name)
    
    if not project_dir.exists():
        raise ValueError(f"Project not found: {project_name}")
    
    # Find the latest reward model
    reward_models_dir = project_dir / 'reward_training'
    if not reward_models_dir.exists():
        raise ValueError(f"No reward training found in project: {project_name}")
    
    # Load the most recent model
    model_files = list(reward_models_dir.glob('best_model_*.pth'))
    if not model_files:
        raise ValueError(f"No trained models found in: {reward_models_dir}")
    
    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
    
    # Find encoder
    encoder_path = project_dir / 'models' / 'base_models' / 'action_encoder.pkl'
    if not encoder_path.exists():
        # Try alternative location
        encoder_path = project_dir / 'experiments' / 'base_model' / 'models' / 'action_encoder.pkl'
    
    if not encoder_path.exists():
        raise ValueError(f"Encoder not found for project: {project_name}")
    
    # Initialize agent
    agent = RewardBasedExecutionAgent(str(latest_model), str(encoder_path))
    smart_agent = SmartUIAgent(agent, project_name)
    
    print(f"✅ Loaded project: {project_name}")
    print(f"🤖 Model: {latest_model.name}")
    
    return smart_agent

# Example usage functions
def example_login_automation():
    """Complete example: Login automation from start to finish"""
    print("🔑 EXAMPLE: Complete Login Automation Pipeline")
    print("=" * 50)
    
    # Your data files (replace with actual paths)
    data_files = [
        "data/login_gmail.json",
        "data/login_facebook.json", 
        # Add your JSON files here
    ]
    
    # Create complete project
    agent = create_new_project(
        project_name="smart_login_bot",
        data_files=data_files,
        task_type='login'
    )
    
    # Test the agent
    print("\n🧪 Testing the smart agent...")
    
    # Predict next action
    prediction = agent.predict_next_action()
    print(f"Next action prediction: {prediction}")
    
    # Run automation (uncomment to actually execute)
    # agent.automate_task('login_test', max_actions=15)
    
    # Continuous learning (uncomment to run more learning)
    # agent.continuous_learning_mode('login_improvement', learning_episodes=5)
    
    return agent

def example_multi_task_automation():
    """Example: Multi-task automation system"""
    print("🎯 EXAMPLE: Multi-Task Automation System")
    print("=" * 45)
    
    # Different task types
    tasks = {
        'login': ["data/login_data.json"],
        'search': ["data/search_data.json"],
        'form_fill': ["data/form_data.json"]
    }
    
    agents = {}
    
    for task_type, data_files in tasks.items():
        print(f"\n📋 Creating {task_type} automation...")
        
        agent = create_new_project(
            project_name=f"smart_{task_type}_bot",
            data_files=data_files,
            task_type=task_type
        )
        
        agents[task_type] = agent
    
    print(f"\n🎉 Created {len(agents)} specialized automation agents!")
    return agents

def quick_start_guide():
    """Quick start guide for new users"""
    guide = """
🚀 UI AUTOMATION QUICK START GUIDE
==================================

1. PREPARE YOUR DATA
   - Put your JSON files (like the one you showed) in a folder
   - Each file should have 'screenshots' and 'actions'

2. CREATE PROJECT
   ```python
   from complete_integration import create_new_project
   
   agent = create_new_project(
       project_name="my_automation_bot",
       data_files=["path/to/your/data.json"], 
       task_type='login'  # or 'search', 'form_fill'
   )
   ```

3. USE YOUR AGENT
   ```python
   # Predict what to do next
   action = agent.predict_next_action()
   print(action)
   
   # Run full automation
   agent.automate_task('login', confidence_threshold=0.8)
   
   # Continuous learning
   agent.continuous_learning_mode('login_practice', learning_episodes=10)
   ```

4. LOAD EXISTING PROJECT
   ```python
   from complete_integration import load_project
   agent = load_project("my_automation_bot")
   ```

🎯 REWARD SYSTEM FEATURES:
- ✅ +100 points for successful task completion
- ✅ +10 points for each meaningful progress step
- ✅ +5 points for clicking correct UI elements
- ❌ -5 points for clicking wrong elements
- ❌ -1 point per action (encourages efficiency)
- 🏆 Bonus points for speed and accuracy

🧠 THE AGENT LEARNS TO:
- Complete tasks faster
- Click more precisely
- Avoid wrong elements
- Adapt to new scenarios
- Improve over time

🚀 READY TO START? Run:
python complete_integration.py
"""
    
    print(guide)
    return guide

# Main execution
def main():
    """Main function demonstrating the complete system"""
    print("🤖 UI AUTOMATION SYSTEM - COMPLETE INTEGRATION")
    print("=" * 55)
    
    # Show quick start guide
    quick_start_guide()
    
    # Ask user what they want to do
    print("\nWhat would you like to do?")
    print("1. Create new login automation project")
    print("2. Create multi-task automation system") 
    print("3. Load existing project")
    print("4. Show examples only")
    
    try:
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            # Simple login automation
            print("\n🔑 Creating login automation project...")
            
            # Get data files from user
            data_files = input("Enter path to your JSON data files (comma-separated): ").strip().split(',')
            data_files = [f.strip() for f in data_files if f.strip()]
            
            if not data_files:
                print("Using example data files...")
                data_files = ["data/sample_login.json"]  # Use sample data
            
            agent = create_new_project("my_login_bot", data_files, 'login')
            
            # Quick test
            print("\n🧪 Quick test...")
            prediction = agent.predict_next_action()
            print(f"Agent predicts: {prediction}")
            
        elif choice == '2':
            example_multi_task_automation()
            
        elif choice == '3':
            project_name = input("Enter project name to load: ").strip()
            agent = load_project(project_name)
            print(f"✅ Loaded project: {project_name}")
            
        elif choice == '4':
            print("📚 Showing examples...")
            print("\nLogin automation example:")
            print("agent = create_new_project('login_bot', ['data.json'], 'login')")
            print("\nMulti-task example:")
            print("agents = example_multi_task_automation()")
            
        else:
            print("Invalid choice. Showing quick start guide.")
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Try running the quick start guide first")

if __name__ == "__main__":
    main()
"""
from complete_integration import create_new_project

# Your JSON data files
data_files = ["login_task_1.json", "login_task_2.json"]

# Create complete smart agent
agent = create_new_project(
    project_name="smart_login_bot",
    data_files=data_files,
    task_type='login'
)

# Now use it!
agent.automate_task('login', confidence_threshold=0.8)
"""