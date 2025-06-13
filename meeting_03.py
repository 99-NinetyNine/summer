import time
import json
import threading
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pyautogui
import pynput
from pynput import mouse, keyboard
from PIL import Image
import os
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict, Any

@dataclass
class ActionEvent:
    """Single action event with image filenames"""
    timestamp: float
    timestamp_ms: int
    action_type: str  # 'click', 'double_click', 'scroll', 'key_press', 'text_input'
    image_files: List[str]  # Filenames of 5 screenshots around the event
    screen_width: int
    screen_height: int
    
    # Action-specific data
    x: Optional[int] = None
    y: Optional[int] = None
    x1: Optional[int] = None  # For scroll start
    y1: Optional[int] = None
    x2: Optional[int] = None  # For scroll end
    y2: Optional[int] = None
    scroll_dx: Optional[int] = None
    scroll_dy: Optional[int] = None
    button: Optional[str] = None  # 'left', 'right', 'middle'
    key: Optional[str] = None
    text: Optional[str] = None

@dataclass 
class TrainingSample:
    """Complete training sample for one recording session"""
    session_id: str
    task_label: str
    events: List[ActionEvent]
    total_duration: float
    screen_resolution: Tuple[int, int]
    created_at: str
    images_dir: str  # Directory containing all screenshot files

class UIDataRecorder:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("UI Automation Data Recorder")
        self.root.geometry("500x600")
        self.root.configure(bg="#2b2b3b")
        
        # Recording state
        self.recording = False
        self.start_time = None
        self.countdown_duration = 3  # configurable
        self.current_session = None
        self.events = []
        self.session_id = None
        self.task_label = ""
        
        # Event listeners
        self.mouse_listener = None
        self.keyboard_listener = None
        
        # Screenshot management
        self.screenshot_buffer = []
        self.buffer_size = 20  # Keep last 20 screenshots
        self.screenshot_thread = None
        self.screenshot_lock = threading.Lock()
        self.screenshot_counter = 0
        self.images_dir = None
        
        # Text input tracking
        self.text_buffer = ""
        self.last_key_time = 0
        self.text_timeout = 1.0  # Group keys within 1 second as text input
        self.pressed_keys = set()
        
        # Get screen resolution
        self.screen_width = pyautogui.size().width
        self.screen_height = pyautogui.size().height
        
        # Setup UI
        self.setup_ui()
        print(f"Screen resolution: {self.screen_width}x{self.screen_height}")
        
    def setup_ui(self):
        """Setup the GUI interface"""
        # Style
        style = ttk.Style()
        style.theme_use("clam")
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="🎬 UI Automation Data Recorder", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Task label input
        ttk.Label(main_frame, text="Task Label:").pack(anchor="w")
        self.task_var = tk.StringVar()
        task_entry = ttk.Entry(main_frame, textvariable=self.task_var, width=50)
        task_entry.pack(fill="x", pady=(5, 15))
        task_entry.insert(0, "login_to_website")
        
        # Configuration
        config_frame = ttk.LabelFrame(main_frame, text="Configuration", padding="10")
        config_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Label(config_frame, text="Countdown Duration (seconds):").grid(row=0, column=0, sticky=tk.W)
        self.countdown_var = tk.StringVar(value="3")
        countdown_entry = ttk.Entry(config_frame, textvariable=self.countdown_var, width=10)
        countdown_entry.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        
        # Status
        self.status_var = tk.StringVar(value="Ready to Record")
        status_label = ttk.Label(main_frame, textvariable=self.status_var, font=('Arial', 12))
        status_label.pack(pady=(10, 5))
        
        self.session_label = ttk.Label(main_frame, text="No active session", 
                                     font=('Arial', 10), foreground='gray')
        self.session_label.pack()
        
        # Timer
        self.timer_var = tk.StringVar(value="00:00")
        timer_label = ttk.Label(main_frame, textvariable=self.timer_var, 
                               font=('Arial', 20, 'bold'))
        timer_label.pack(pady=(10, 20))
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=10)
        
        self.start_btn = ttk.Button(button_frame, text="▶️ START RECORDING", 
                                   command=self.start_recording, width=20)
        self.start_btn.pack(side="left", padx=5)
        
        self.stop_btn = ttk.Button(button_frame, text="⏹️ STOP RECORDING", 
                                  command=self.stop_recording, width=20, state="disabled")
        self.stop_btn.pack(side="left", padx=5)
        
        # Statistics
        stats_frame = ttk.LabelFrame(main_frame, text="Session Stats", padding="10")
        stats_frame.pack(fill="x", pady=20)
        
        self.stats_text = tk.Text(stats_frame, height=8, width=50, state="disabled")
        self.stats_text.pack(fill="both", expand=True)
        
        # Data management
        action_frame = ttk.Frame(main_frame)
        action_frame.pack(pady=10)
        
        ttk.Button(action_frame, text="💾 Save Episode", 
                  command=self.save_episode, width=15).pack(side="left", padx=5)
        ttk.Button(action_frame, text="📁 View Sessions", 
                  command=self.view_sessions, width=15).pack(side="left", padx=5)
        ttk.Button(action_frame, text="🗑️ Clear All", 
                  command=self.clear_data, width=15).pack(side="left", padx=5)
        
        # Instructions
        instructions = """Instructions:
1. Enter task label (e.g., 'login_to_facebook')
2. Click START → countdown → move to target app
3. Listen for bell sound → start your actions
4. Press Ctrl+Shift+K when done (this hotkey won't be recorded)"""
        
        instructions_label = ttk.Label(main_frame, text=instructions, 
                                     font=('Arial', 9), justify=tk.LEFT)
        instructions_label.pack(pady=10)
        
        self.update_stats()
    
    def start_recording(self):
        """Start the recording process with countdown"""
        if not self.task_var.get().strip():
            messagebox.showerror("Error", "Please enter a task label!")
            return
            
        try:
            self.countdown_duration = int(self.countdown_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid countdown duration")
            return
        
        self.task_label = self.task_var.get().strip()
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        
        # Generate session ID and create directories
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.events = []
        self.screenshot_counter = 0
        
        # Create directories
        data_dir = Path("training_data")
        data_dir.mkdir(exist_ok=True)
        self.images_dir = data_dir / self.session_id / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        self.session_label.config(text=f"Session: {self.session_id}")
        
        # Start countdown in separate thread
        threading.Thread(target=self.countdown_and_start, daemon=True).start()
    
    def countdown_and_start(self):
        """Countdown and then start actual recording"""
        for i in range(self.countdown_duration, 0, -1):
            self.root.after(0, lambda i=i: self.status_var.set(f"Starting in {i}..."))
            time.sleep(1)
        
        # Play bell sounds and start recording
        self.root.after(0, lambda: self.status_var.set("🔴 RECORDING! (Ctrl+Shift+K to stop)"))
        
        # Bell sounds
        for _ in range(3):
            print('\a')
            time.sleep(0.2)
        
        # Start recording
        self.recording = True
        self.start_time = time.time()
        self.start_screenshot_capture()
        self.start_listeners()
        
        # Start timer update
        self.root.after(0, self.update_timer)
        
        print(f"🎬 Recording started: {self.task_label}")
    
    def start_screenshot_capture(self):
        """Start continuous screenshot capture in background"""
        def capture_screenshots():
            while self.recording:
                try:
                    screenshot = pyautogui.screenshot()
                    screenshot_array = np.array(screenshot)
                    
                    with self.screenshot_lock:
                        self.screenshot_buffer.append({
                            'image': screenshot_array,
                            'timestamp': time.time()
                        })
                        
                        # Keep only last N screenshots
                        if len(self.screenshot_buffer) > self.buffer_size:
                            self.screenshot_buffer.pop(0)
                    
                    time.sleep(0.02)  # 20ms between screenshots
                    
                except Exception as e:
                    print(f"Screenshot error: {e}")
                    time.sleep(0.1)
        
        self.screenshot_thread = threading.Thread(target=capture_screenshots, daemon=True)
        self.screenshot_thread.start()
    
    def start_listeners(self):
        """Start event listeners using pynput"""
        # Mouse listener
        self.mouse_listener = mouse.Listener(
            on_click=self.on_mouse_click,
            on_scroll=self.on_mouse_scroll
        )
        self.mouse_listener.start()
        
        # Keyboard listener  
        self.keyboard_listener = keyboard.Listener(
            on_press=self.on_key_press,
            on_release=self.on_key_release
        )
        self.keyboard_listener.start()
        
        print("Event listeners started")
    
    def stop_listeners(self):
        """Stop event listeners"""
        if self.mouse_listener:
            self.mouse_listener.stop()
        if self.keyboard_listener:
            self.keyboard_listener.stop()
    
    def get_timestamp_ms(self):
        """Get timestamp in milliseconds from start"""
        return int((time.time() - self.start_time) * 1000) if self.start_time else 0
    
    def save_screenshots_around_event(self, event_time: float) -> List[str]:
        """Save 5 screenshots around the event time and return filenames"""
        with self.screenshot_lock:
            if not self.screenshot_buffer:
                return []
            
            # Find screenshots closest to event time
            screenshots = []
            for item in self.screenshot_buffer:
                time_diff = abs(item['timestamp'] - event_time)
                if time_diff <= 0.2:  # Within 200ms
                    screenshots.append((item['image'], time_diff))
            
            # Sort by time difference and take up to 5
            screenshots.sort(key=lambda x: x[1])
            selected_screenshots = screenshots[:5]
            
            # If we don't have 5, pad with the most recent
            while len(selected_screenshots) < 5 and self.screenshot_buffer:
                selected_screenshots.append((self.screenshot_buffer[-1]['image'], 0))
            
            # Save screenshots and return filenames
            filenames = []
            for i, (img_array, _) in enumerate(selected_screenshots[:5]):
                self.screenshot_counter += 1
                filename = f"screenshot_{self.screenshot_counter:06d}.png"
                filepath = self.images_dir / filename
                
                # Convert numpy array to PIL Image and save
                img = Image.fromarray(img_array)
                img.save(filepath, 'PNG')
                filenames.append(filename)
            
            return filenames
    
    def on_mouse_click(self, x, y, button, pressed):
        """Handle mouse click events"""
        if not self.recording:
            return
            
        # Only record press events (not release)
        if not pressed:
            return
            
        try:
            event_time = time.time()
            image_files = self.save_screenshots_around_event(event_time)
            
            # Determine click type (simple heuristic - could be improved)
            action_type = 'click'
            
            event = ActionEvent(
                timestamp=event_time,
                timestamp_ms=self.get_timestamp_ms(),
                action_type=action_type,
                image_files=image_files,
                screen_width=self.screen_width,
                screen_height=self.screen_height,
                x=x,
                y=y,
                button=str(button).split('.')[-1]  # left, right, middle
            )
            
            self.events.append(event)
            self.update_event_count()
            
            print(f"🖱️  Mouse {action_type}: {button} at ({x}, {y}) - {event.timestamp_ms}ms")
            
        except Exception as e:
            print(f"Mouse click error: {e}")
    
    def on_mouse_scroll(self, x, y, dx, dy):
        """Handle mouse scroll events"""
        if not self.recording:
            return
            
        try:
            event_time = time.time()
            image_files = self.save_screenshots_around_event(event_time)
            
            # Determine scroll direction
            if dy > 0:
                action_type = 'scroll_up'
            elif dy < 0:
                action_type = 'scroll_down'
            elif dx > 0:
                action_type = 'scroll_right'
            elif dx < 0:
                action_type = 'scroll_left'
            else:
                action_type = 'scroll'
            
            event = ActionEvent(
                timestamp=event_time,
                timestamp_ms=self.get_timestamp_ms(),
                action_type=action_type,
                image_files=image_files,
                screen_width=self.screen_width,
                screen_height=self.screen_height,
                x=x,
                y=y,
                scroll_dx=dx,
                scroll_dy=dy
            )
            
            self.events.append(event)
            self.update_event_count()
            
            print(f"🖱️  Mouse scroll: {action_type} ({dx}, {dy}) at ({x}, {y}) - {event.timestamp_ms}ms")
            
        except Exception as e:
            print(f"Mouse scroll error: {e}")
    
    def on_key_press(self, key):
        """Handle keyboard press events"""
        if not self.recording:
            return
            
        try:
            # Get key name
            key_name = key.char if hasattr(key, 'char') and key.char else str(key).split('.')[-1]
            self.pressed_keys.add(key_name.lower())
            
            # Check for stop recording hotkey (Ctrl+Shift+K)
            if {'ctrl_l', 'shift', 'k'}.issubset(self.pressed_keys) or \
               {'ctrl_r', 'shift', 'k'}.issubset(self.pressed_keys):
                print("🚀 Ctrl+Shift+K detected! Stopping recording...")
                self.pressed_keys.clear()
                self.root.after(0, self.stop_recording)
                return
            
            current_time = time.time()
            
            # Handle text input vs single key press
            if hasattr(key, 'char') and key.char and key.char.isprintable():
                # Printable character - part of text input
                self.text_buffer += key.char
                self.last_key_time = current_time
                
                # Set timer to process text input after timeout
                threading.Timer(self.text_timeout, self.process_text_input).start()
                
            else:
                # Single key press (Enter, Escape, etc.)
                self.flush_text_buffer()  # Save any pending text first
                
                image_files = self.save_screenshots_around_event(current_time)
                
                event = ActionEvent(
                    timestamp=current_time,
                    timestamp_ms=self.get_timestamp_ms(),
                    action_type='key_press',
                    image_files=image_files,
                    screen_width=self.screen_width,
                    screen_height=self.screen_height,
                    key=key_name
                )
                
                self.events.append(event)
                self.update_event_count()
                
                print(f"⌨️  Key press: {key_name} - {event.timestamp_ms}ms")
                
        except Exception as e:
            print(f"Keyboard press error: {e}")
    
    def on_key_release(self, key):
        """Handle keyboard release events"""
        if not self.recording:
            return
            
        try:
            key_name = key.char if hasattr(key, 'char') and key.char else str(key).split('.')[-1]
            self.pressed_keys.discard(key_name.lower())
        except:
            pass
    
    def process_text_input(self):
        """Process accumulated text input after timeout"""
        if not self.recording:
            return
            
        current_time = time.time()
        
        # Only process if enough time has passed since last key
        if current_time - self.last_key_time >= self.text_timeout and self.text_buffer:
            self.flush_text_buffer()
    
    def flush_text_buffer(self):
        """Save accumulated text as text_input event"""
        if self.text_buffer and self.recording:
            try:
                event_time = time.time()
                image_files = self.save_screenshots_around_event(event_time)
                
                event = ActionEvent(
                    timestamp=event_time,
                    timestamp_ms=self.get_timestamp_ms(),
                    action_type='text_input',
                    image_files=image_files,
                    screen_width=self.screen_width,
                    screen_height=self.screen_height,
                    text=self.text_buffer
                )
                
                self.events.append(event)
                self.update_event_count()
                
                print(f"⌨️  Text input: '{self.text_buffer}' - {event.timestamp_ms}ms")
                
            except Exception as e:
                print(f"Text input error: {e}")
            
            finally:
                self.text_buffer = ""
    
    def update_event_count(self):
        """Update UI with current event count"""
        count = len(self.events)
        self.root.after(0, lambda: self.status_var.set(
            f"🔴 RECORDING... ({count} events captured)"))
    
    def update_timer(self):
        """Update the timer display"""
        if self.recording and self.start_time:
            elapsed = time.time() - self.start_time
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            self.timer_var.set(f"{minutes:02d}:{seconds:02d}")
            self.root.after(1000, self.update_timer)
    
    def stop_recording(self):
        """Stop the recording process"""
        if not self.recording:
            return
            
        self.recording = False
        end_time = time.time()
        
        # Flush any remaining text
        self.flush_text_buffer()
        
        # Stop listeners
        self.stop_listeners()
        
        # Update UI
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        
        # Update stats
        duration = end_time - self.start_time if self.start_time else 0
        
        if self.events:
            self.status_var.set(f"⏹️ Recording stopped. {len(self.events)} events captured.")
            print(f"⏹️ Recording stopped. Duration: {duration:.1f}s, Events: {len(self.events)}")
        else:
            self.status_var.set("⏹️ Recording stopped. No events captured.")
        
        self.update_stats()
    
    def save_episode(self):
        """Save the recording session to disk"""
        if not self.events:
            messagebox.showwarning("Warning", "No data to save!")
            return
            
        try:
            # Create session object
            duration = self.events[-1].timestamp - self.events[0].timestamp if self.events else 0
            
            session = TrainingSample(
                session_id=self.session_id,
                task_label=self.task_label,
                events=self.events,
                total_duration=duration,
                screen_resolution=(self.screen_width, self.screen_height),
                created_at=datetime.now().isoformat(),
                images_dir=str(self.images_dir)
            )
            
            # Save session data as JSON
            data_dir = Path("training_data")
            session_file = data_dir / f"{self.session_id}.json"
            
            # Convert to dict for JSON serialization
            session_dict = asdict(session)
            
            with open(session_file, 'w') as f:
                json.dump(session_dict, f, indent=2)
            
            messagebox.showinfo("Success", f"Episode saved to {session_file}")
            print(f"💾 Episode saved: {session_file}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save episode: {e}")
    
    def view_sessions(self):
        """View saved recording sessions"""
        data_dir = Path("training_data")
        if not data_dir.exists():
            messagebox.showinfo("Info", "No training data found")
            return
        
        # Get all session files
        session_files = list(data_dir.glob("*.json"))
        
        if not session_files:
            messagebox.showinfo("Info", "No sessions found")
            return
        
        # Create sessions window
        sessions_window = tk.Toplevel(self.root)
        sessions_window.title("Recorded Sessions")
        sessions_window.geometry("700x500")
        
        # Sessions list
        frame = ttk.Frame(sessions_window, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(frame, text="Recorded Sessions", font=('Arial', 14, 'bold')).pack(pady=(0, 10))
        
        # Treeview for sessions
        columns = ('Session ID', 'Task', 'Events', 'Duration', 'Created')
        tree = ttk.Treeview(frame, columns=columns, show='headings', height=15)
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=120)
        
        tree.pack(fill=tk.BOTH, expand=True)
        
        # Populate sessions list
        for session_file in sorted(session_files):
            try:
                with open(session_file, 'r') as f:
                    session_data = json.load(f)
                
                tree.insert('', 'end', values=(
                    session_data.get('session_id', ''),
                    session_data.get('task_label', ''),
                    len(session_data.get('events', [])),
                    f"{session_data.get('total_duration', 0):.1f}s",
                    session_data.get('created_at', '')[:19]
                ))
            except Exception as e:
                print(f"Error loading session {session_file}: {e}")
    
    def clear_data(self):
        """Clear all recorded data"""
        if messagebox.askyesno("Confirm", "Are you sure you want to delete all recorded data?"):
            try:
                data_dir = Path("training_data")
                if data_dir.exists():
                    import shutil
                    shutil.rmtree(data_dir)
                
                messagebox.showinfo("Success", "All data cleared")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to clear data: {e}")
    
    def update_stats(self):
        """Update statistics display"""
        if not hasattr(self, 'events'):
            self.events = []
            
        # Count event types
        mouse_clicks = len([e for e in self.events if e.action_type == 'click'])
        key_presses = len([e for e in self.events if e.action_type == 'key_press'])
        text_inputs = len([e for e in self.events if e.action_type == 'text_input'])
        scrolls = len([e for e in self.events if 'scroll' in e.action_type])
        
        duration = 0
        if self.events and len(self.events) > 1:
            duration = self.events[-1].timestamp - self.events[0].timestamp
        
        stats = f"""Task: {getattr(self, 'task_label', 'None')}
Session: {getattr(self, 'session_id', 'None')}
Duration: {duration:.1f}s
Total Events: {len(self.events)}
Images Captured: {len(self.events) * 5}

Event Breakdown:
- Mouse clicks: {mouse_clicks}
- Key presses: {key_presses}  
- Text inputs: {text_inputs}
- Scroll events: {scrolls}

Screen Size: {self.screen_width}x{self.screen_height}"""
        
        self.stats_text.config(state="normal")
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats)
        self.stats_text.config(state="disabled")
    
    def run(self):
        """Run the application"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
    
    def on_closing(self):
        """Handle application closing"""
        if self.recording:
            self.stop_recording()
        
        self.root.destroy()

if __name__ == "__main__":
    print("🚀 Starting UI Data Recorder...")
    print("📋 Instructions:")
    print("1. Enter a task label (e.g., 'login_to_facebook')")
    print("2. Click START to begin recording")
    print("3. Wait for countdown and bell sound")
    print("4. Perform your UI actions")
    print("5. Press Ctrl+Shift+K when finished")
    print()
    
    # Check dependencies
    try:
        import pyautogui
        import pynput
        from PIL import Image
        print("✅ All dependencies available")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Install with: pip install pyautogui pynput pillow numpy")
        exit(1)
    
    app = UIDataRecorder()
    app.run()