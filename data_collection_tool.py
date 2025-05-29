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
import io
import base64


class DesktopActionRecorder:
    def __init__(self):
        self.recording = False
        self.start_time = None
        self.current_episode = None
        self.episode_data = {
            "task_label": "login",
            "start_time": "",
            "end_time": "",
            "duration_ms": 0,
            "screen_size": {"width": 0, "height": 0},
            "screenshots": [],
            "actions": [],
        }

        # Event listeners
        self.mouse_listener = None
        self.keyboard_listener = None
        self.screenshot_thread = None
        self.screenshot_interval = 1.0  # seconds

        # shortcuts
        self.pressed_keys = set()
        self.setup_gui()

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Desktop Action Recorder")
        self.root.geometry("500x600")
        self.root.configure(bg="#2b2b3b")

        # Style
        style = ttk.Style()
        style.theme_use("clam")

        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(
            main_frame, text="🎬 Desktop Action Recorder", font=("Arial", 16, "bold")
        )
        title_label.pack(pady=(0, 20))

        # Task label input
        ttk.Label(main_frame, text="Task Label:").pack(anchor="w")
        self.task_var = tk.StringVar()
        task_entry = ttk.Entry(main_frame, textvariable=self.task_var, width=50)
        task_entry.pack(fill="x", pady=(5, 15))
        task_entry.insert(0, "login_to_website")

        # Status
        self.status_var = tk.StringVar(value="Ready to Record")
        status_label = ttk.Label(
            main_frame, textvariable=self.status_var, font=("Arial", 12)
        )
        status_label.pack(pady=(0, 10))

        # Timer
        self.timer_var = tk.StringVar(value="00:00")
        timer_label = ttk.Label(
            main_frame, textvariable=self.timer_var, font=("Arial", 20, "bold")
        )
        timer_label.pack(pady=(0, 20))

        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=10)

        self.start_btn = ttk.Button(
            button_frame, text="▶️ START", command=self.start_recording, width=15
        )
        self.start_btn.pack(side="left", padx=5)

        self.stop_btn = ttk.Button(
            button_frame,
            text="⏹️ STOP",
            command=self.stop_recording,
            width=15,
            state="disabled",
        )
        self.stop_btn.pack(side="left", padx=5)

        # Settings
        settings_frame = ttk.LabelFrame(main_frame, text="Settings", padding="10")
        settings_frame.pack(fill="x", pady=20)

        ttk.Label(settings_frame, text="Screenshot Interval (seconds):").pack(
            anchor="w"
        )
        self.interval_var = tk.DoubleVar(value=1.0)
        interval_spin = ttk.Spinbox(
            settings_frame,
            from_=0.5,
            to=5.0,
            increment=0.5,
            textvariable=self.interval_var,
            width=10,
        )
        interval_spin.pack(anchor="w", pady=5)

        # Statistics
        stats_frame = ttk.LabelFrame(main_frame, text="Session Stats", padding="10")
        stats_frame.pack(fill="x", pady=10)

        self.stats_text = tk.Text(stats_frame, height=8, width=50, state="disabled")
        self.stats_text.pack(fill="both", expand=True)

        # Action buttons
        action_frame = ttk.Frame(main_frame)
        action_frame.pack(pady=10)

        ttk.Button(
            action_frame, text="💾 Save Episode", command=self.save_episode, width=15
        ).pack(side="left", padx=5)
        ttk.Button(
            action_frame, text="📁 Load Episode", command=self.load_episode, width=15
        ).pack(side="left", padx=5)
        ttk.Button(
            action_frame, text="🗑️ Clear", command=self.clear_data, width=15
        ).pack(side="left", padx=5)

        self.update_stats()

    def start_recording(self):
        if not self.task_var.get().strip():
            messagebox.showerror("Error", "Please enter a task label!")
            return
        self.status_var.set("About to start...")
        time.sleep(3)
        print('\a')  # Bell character - works in most terminals
        print('\a')  # Bell character - works in most terminals
        print('\a')  # Bell character - works in most terminals
        print("Listeners started")

        self.recording = True
        self.start_time = time.time()
        self.screenshot_interval = self.interval_var.get()

        # Initialize episode data
        self.episode_data = {
            "task_label": self.task_var.get().strip(),
            "start_time": datetime.now().isoformat(),
            "end_time": "",
            "duration_ms": 0,
            "screen_size": {
                "width": pyautogui.size()[0],
                "height": pyautogui.size()[1],
            },
            "screenshots": [],
            "actions": [],
        }

        # Update UI
        self.status_var.set("🔴 RECORDING...")
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")

        # Start listeners
        
        self.start_listeners()

        # Start screenshot thread
        self.screenshot_thread = threading.Thread(target=self.capture_screenshots)
        self.screenshot_thread.daemon = True
        self.screenshot_thread.start()

        # Start timer update
        self.update_timer()

        print(f"🎬 Recording started: {self.episode_data['task_label']}")

    def stop_recording(self):
        self.recording = False
        end_time = time.time()

        # Update episode data
        self.episode_data["end_time"] = datetime.now().isoformat()
        self.episode_data["duration_ms"] = int((end_time - self.start_time) * 1000)

        # Stop listeners
        self.stop_listeners()

        # Update UI
        self.status_var.set("⏹️ Recording Stopped")
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

        self.update_stats()
        print(
            f"⏹️ Recording stopped. Duration: {self.episode_data['duration_ms']/1000:.1f}s"
        )

    def start_listeners(self):
        # Mouse listener
        self.mouse_listener = mouse.Listener(
            on_click=self.on_mouse_click, on_scroll=self.on_mouse_scroll
        )
        self.mouse_listener.start()

        # Keyboard listener
        self.keyboard_listener = keyboard.Listener(
            on_press=self.on_key_press, on_release=self.on_key_release
        )
        self.keyboard_listener.start()

    def stop_listeners(self):
        if self.mouse_listener:
            self.mouse_listener.stop()
        if self.keyboard_listener:
            self.keyboard_listener.stop()

    def get_timestamp(self):
        return int((time.time() - self.start_time) * 1000) if self.start_time else 0

    def on_mouse_click(self, x, y, button, pressed):
        if not self.recording:
            return

        action = {
            "timestamp_ms": self.get_timestamp(),
            "type": "mouse",
            "action": "click" if pressed else "release",
            "button": str(button).split(".")[-1],  # left, right, middle
            "coordinates": {"x": x, "y": y},
            "screen_size": {
                "width": pyautogui.size()[0],
                "height": pyautogui.size()[1],
            },
        }

        self.episode_data["actions"].append(action)
        print(
            f"🖱️  Mouse {action['action']}: {button} at ({x}, {y}) - {action['timestamp_ms']}ms"
        )

    def on_mouse_scroll(self, x, y, dx, dy):
        if not self.recording:
            return

        action = {
            "timestamp_ms": self.get_timestamp(),
            "type": "mouse",
            "action": "scroll",
            "coordinates": {"x": x, "y": y},
            "scroll": {"dx": dx, "dy": dy},
        }

        self.episode_data["actions"].append(action)
        print(
            f"🖱️  Mouse scroll: ({dx}, {dy}) at ({x}, {y}) - {action['timestamp_ms']}ms"
        )

    def on_key_press(self, key):
        if not self.recording:
            return

        try:
            key_name = (
                key.char
                if hasattr(key, "char") and key.char
                else str(key).split(".")[-1]
            )
        except:
            key_name = str(key).split(".")[-1]
        self.pressed_keys.add(key_name.lower())
        # Check for Ctrl + Shift + K
        if {"ctrl", "shift", "k"}.issubset(self.pressed_keys):
            print("🚀 Ctrl + Shift + K detected! Triggering special action...")
            self.pressed_keys = set()
            self.stop_recording()
            return 
        
        action = {
            "timestamp_ms": self.get_timestamp(),
            "type": "keyboard",
            "action": "press",
            "key": key_name,
            "key_code": getattr(key, "vk", None) if hasattr(key, "vk") else None,
        }

        self.episode_data["actions"].append(action)
        print(f"⌨️  Key press: {key_name} - {action['timestamp_ms']}ms")

    def on_key_release(self, key):
        if not self.recording:
            return

        try:
            key_name = (
                key.char
                if hasattr(key, "char") and key.char
                else str(key).split(".")[-1]
            )
        except:
            key_name = str(key).split(".")[-1]
        self.pressed_keys  = set()

        action = {
            "timestamp_ms": self.get_timestamp(),
            "type": "keyboard",
            "action": "release",
            "key": key_name,
        }

        self.episode_data["actions"].append(action)

    def capture_screenshots(self):
        while self.recording:
            try:
                # Capture screenshot
                screenshot = pyautogui.screenshot()

                # Convert to base64 for storage
                buffer = io.BytesIO()
                screenshot.save(buffer, format="PNG")
                img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

                screenshot_data = {
                    "timestamp_ms": self.get_timestamp(),
                    "image_base64": img_base64,
                    "size": {"width": screenshot.width, "height": screenshot.height},
                }

                self.episode_data["screenshots"].append(screenshot_data)
                print(f"📸 Screenshot captured - {screenshot_data['timestamp_ms']}ms")

            except Exception as e:
                print(f"Screenshot error: {e}")

            time.sleep(self.screenshot_interval)

    def update_timer(self):
        if self.recording and self.start_time:
            elapsed = time.time() - self.start_time
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            self.timer_var.set(f"{minutes:02d}:{seconds:02d}")
            self.root.after(1000, self.update_timer)

    def update_stats(self):
        stats = f"""Task: {self.episode_data.get('task_label', 'login')}
Duration: {self.episode_data.get('duration_ms', 0)/1000:.1f}s
Total Actions: {len(self.episode_data.get('actions', []))}
Screenshots: {len(self.episode_data.get('screenshots', []))}
Screen Size: {self.episode_data.get('screen_size', {}).get('width', 0)}x{self.episode_data.get('screen_size', {}).get('height', 0)}

Action Breakdown:
- Mouse clicks: {len([a for a in self.episode_data.get('actions', []) if a.get('type') == 'mouse' and a.get('action') == 'click'])}
- Key presses: {len([a for a in self.episode_data.get('actions', []) if a.get('type') == 'keyboard' and a.get('action') == 'press'])}
- Scroll events: {len([a for a in self.episode_data.get('actions', []) if a.get('type') == 'mouse' and a.get('action') == 'scroll'])}"""

        self.stats_text.config(state="normal")
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats)
        self.stats_text.config(state="disabled")

    def save_episode(self):
        if not self.episode_data.get("actions"):
            messagebox.showwarning("Warning", "No data to save!")
            return

        filename = filedialog.asksaveasfilename(
            confirmoverwrite=True,
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            
            initialfile=f"episode_{self.episode_data['task_label']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )

        if filename:
            with open(filename, "w") as f:
                json.dump(self.episode_data, f, indent=2)
            messagebox.showinfo("Success", f"Episode saved to {filename}")
            print(f"💾 Episode saved: {filename}")

    def load_episode(self):
        filename = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])

        if filename:
            try:
                with open(filename, "r") as f:
                    self.episode_data = json.load(f)
                self.task_var.set(self.episode_data.get("task_label", ""))
                self.update_stats()
                messagebox.showinfo("Success", f"Episode loaded from {filename}")
                print(f"📁 Episode loaded: {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load episode: {e}")

    def clear_data(self):
        self.episode_data = {
            "task_label": "login",
            "start_time": "",
            "end_time": "",
            "duration_ms": 0,
            "screen_size": {"width": 0, "height": 0},
            "screenshots": [],
            "actions": [],
        }
        self.task_var.set("")
        self.update_stats()
        print("🗑️ Data cleared")

    def run(self):
        try:
            self.root.mainloop()
        finally:
            self.stop_listeners()


if __name__ == "__main__":
    print("🚀 Starting Desktop Action Recorder...")
    print("📋 Instructions:")
    print("1. Enter a task label (e.g., 'login_to_facebook')")
    print("2. Click START to begin recording")
    print("3. Perform your actions on any application")
    print("4. Click STOP when finished")
    print("5. Save the episode data as JSON")
    print()

    # Check dependencies
    try:
        import pyautogui
        import pynput
        from PIL import Image

        print("✅ All dependencies available")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Install with: pip install pyautogui pynput pillow")
        exit(1)

    recorder = DesktopActionRecorder()
    recorder.run()
