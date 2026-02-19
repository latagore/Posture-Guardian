"""
Posture Guardian Desktop App 🌿
A gentle companion that watches your posture and sends overlay alerts
"""

import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import ttk
import threading
import time
from datetime import datetime
import math
import json
import os
import subprocess
import sys

class PostureGuardian:
    def __init__(self):
        # MediaPipe setup
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # State
        self.good_posture = None
        self.bad_posture_counter = 0
        self.alert_duration = 3  # seconds
        self.alert_delay = 5  # seconds before triggering alert
        self.alert_type = 'sound'  # 'sound' or 'popup'
        self.last_alert_time = 0
        self.monitoring = False
        self.calibrating = False
        self.needs_recalibration = False

        # Per-check enable + threshold (all use a 1-20 scale)
        self.checks = {
            'shoulders':    {'enabled': True, 'threshold': 8},
            'neck':         {'enabled': True, 'threshold': 8},
            'head_forward': {'enabled': True, 'threshold': 8},
            'torso':        {'enabled': True, 'threshold': 8},
            'spine':        {'enabled': True, 'threshold': 8},
        }
        
        # Camera
        self.cap = None
        
        # Calibration file
        self.calibration_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'calibration.json'
        )
        self.posture_issues = []
        self.load_calibration()

        # GUI
        self.root = None
        self.video_label = None
        self.status_label = None
        self.alert_window = None
        
    def calculate_angle(self, p1, p2, p3):
        """Calculate angle between three points"""
        radians = math.atan2(p3.y - p2.y, p3.x - p2.x) - \
                  math.atan2(p1.y - p2.y, p1.x - p2.x)
        angle = abs(radians * 180.0 / math.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle
    
    def save_calibration(self):
        """Save calibration data to file"""
        data = {
            'good_posture': self.good_posture,
            'alert_duration': self.alert_duration,
            'alert_delay': self.alert_delay,
            'alert_type': self.alert_type,
            'checks': self.checks
        }
        with open(self.calibration_file, 'w') as f:
            json.dump(data, f)

    # Fields required in calibration data for it to be valid
    REQUIRED_POSTURE_FIELDS = {'shoulder_slope', 'neck_angle', 'lean_depth', 'torso_lean', 'vertical_distance'}

    def load_calibration(self):
        """Load calibration data from file, gracefully handling missing fields"""
        if os.path.exists(self.calibration_file):
            try:
                with open(self.calibration_file, 'r') as f:
                    data = json.load(f)

                # Load posture data if present
                posture = data.get('good_posture')
                if posture and isinstance(posture, dict):
                    if self.REQUIRED_POSTURE_FIELDS.issubset(posture.keys()):
                        self.good_posture = posture
                        self.monitoring = True
                    else:
                        # Missing fields — prompt recalibration but don't crash
                        self.good_posture = None
                        self.needs_recalibration = True

                # Always load settings (independent of posture validity)
                self.alert_duration = data.get('alert_duration', self.alert_duration)
                self.alert_delay = data.get('alert_delay', self.alert_delay)
                self.alert_type = data.get('alert_type', self.alert_type)
                saved_checks = data.get('checks', {})
                for k in self.checks:
                    if k in saved_checks:
                        if isinstance(saved_checks[k], dict):
                            # New format: merge, keeping defaults for missing keys
                            self.checks[k].update(saved_checks[k])
                        else:
                            # Migrate old format (bool only)
                            self.checks[k]['enabled'] = bool(saved_checks[k])
            except (json.JSONDecodeError, KeyError):
                pass

    def check_posture(self, landmarks):
        """Analyze posture from landmarks"""
        try:
            # Get key landmarks
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            left_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value]
            right_ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value]
            left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
            
            # Calculate shoulder slope
            shoulder_slope = abs(
                math.atan2(right_shoulder.y - left_shoulder.y, 
                          right_shoulder.x - left_shoulder.x) * 180 / math.pi
            )
            
            # Calculate neck angle
            avg_ear = type('obj', (object,), {
                'x': (left_ear.x + right_ear.x) / 2,
                'y': (left_ear.y + right_ear.y) / 2
            })()
            avg_shoulder = type('obj', (object,), {
                'x': (left_shoulder.x + right_shoulder.x) / 2,
                'y': (left_shoulder.y + right_shoulder.y) / 2
            })()
            avg_hip = type('obj', (object,), {
                'x': (left_hip.x + right_hip.x) / 2,
                'y': (left_hip.y + right_hip.y) / 2
            })()
            
            neck_angle = self.calculate_angle(avg_hip, avg_shoulder, avg_ear)
            
            # Leaning forward detection (z-depth)
            # MediaPipe z is depth relative to hips; more negative = closer to camera
            # When leaning forward, shoulders move ahead of hips in depth
            avg_shoulder_z = (left_shoulder.z + right_shoulder.z) / 2
            avg_hip_z = (left_hip.z + right_hip.z) / 2
            lean_depth = avg_shoulder_z - avg_hip_z  # more negative = leaning forward

            # Torso lean angle (degrees from vertical)
            torso_dx = avg_shoulder.x - avg_hip.x
            torso_dy = avg_shoulder.y - avg_hip.y
            torso_lean = abs(math.atan2(torso_dx, -torso_dy) * 180 / math.pi)

            # Vertical compression (distance from hip to ear)
            # Decreases when slouching as spine curves
            vertical_distance = math.sqrt(
                (avg_ear.x - avg_hip.x) ** 2 + (avg_ear.y - avg_hip.y) ** 2
            )

            return {
                'shoulder_slope': shoulder_slope,
                'neck_angle': neck_angle,
                'lean_depth': lean_depth,
                'torso_lean': torso_lean,
                'vertical_distance': vertical_distance
            }
        except:
            return None
    
    def show_alert(self):
        """Show posture alert using the configured alert type"""
        current_time = time.time()
        if current_time - self.last_alert_time < self.alert_duration:
            return

        self.last_alert_time = current_time

        if self.alert_type == 'popup':
            self._show_popup_alert()
        else:
            self._show_sound_alert()

    def _show_sound_alert(self):
        """Play a system alert sound"""
        if sys.platform == 'darwin':
            subprocess.Popen(['afplay', '/System/Library/Sounds/Sosumi.aiff'])

    def _show_popup_alert(self):
        """Show a system-wide overlay popup"""
        if self.alert_window:
            try:
                self.alert_window.destroy()
            except:
                pass

        self.alert_window = tk.Toplevel()
        self.alert_window.title("Posture Alert")
        self.alert_window.attributes('-topmost', True)
        self.alert_window.attributes('-alpha', 0.95)
        self.alert_window.overrideredirect(True)

        screen_w = self.alert_window.winfo_screenwidth()
        screen_h = self.alert_window.winfo_screenheight()
        width, height = 500, 220
        x = (screen_w - width) // 2
        y = (screen_h - height) // 2
        self.alert_window.geometry(f'{width}x{height}+{x}+{y}')
        self.alert_window.configure(bg='#FF3B30')

        frame = tk.Frame(self.alert_window, bg='#FF3B30')
        frame.pack(expand=True, fill='both', padx=30, pady=20)

        tk.Label(frame, text="Check Your Posture!",
                 font=('Helvetica', 24, 'bold'), bg='#FF3B30', fg='white').pack()

        issue_text = ", ".join(self.posture_issues) if self.posture_issues else "Sit up straight"
        tk.Label(frame, text=issue_text, font=('Helvetica', 14),
                 bg='#FF3B30', fg='white', wraplength=440).pack(pady=10)

        self.alert_window.after(int(self.alert_duration * 1000),
                               lambda: self.alert_window.destroy() if self.alert_window else None)
    
    def update_status(self, text, color='white'):
        """Update status label"""
        if self.status_label:
            self.status_label.config(text=text, fg=color)
    
    def calibrate_posture(self):
        """Calibrate good posture"""
        self.calibrating = True
        self.good_posture = None
        self.update_status("📸 Sit with good posture... capturing in 3 seconds!", "yellow")
        
        def finish_calibration():
            time.sleep(3)
            self.calibrating = False
            if self.good_posture:
                self.save_calibration()
                self.update_status("✓ Good posture calibrated! Monitoring...", "lightgreen")
                self.monitoring = True
        
        threading.Thread(target=finish_calibration, daemon=True).start()
    
    def process_frame(self):
        """Process camera frame and check posture"""
        if not self.cap or not self.cap.isOpened():
            return
        
        ret, frame = self.cap.read()
        if not ret:
            return
        
        # Flip frame horizontally for mirror view
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.pose.process(rgb_frame)
        
        # Draw landmarks
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
            
            # Check posture
            posture = self.check_posture(results.pose_landmarks.landmark)
            
            if posture:
                # Calibration mode
                if self.calibrating and not self.good_posture:
                    self.good_posture = posture.copy()
                
                # Monitoring mode
                elif self.monitoring and self.good_posture:
                    shoulder_diff = abs(posture['shoulder_slope'] -
                                      self.good_posture['shoulder_slope'])
                    neck_diff = abs(posture['neck_angle'] -
                                   self.good_posture['neck_angle'])
                    torso_diff = abs(posture['torso_lean'] -
                                    self.good_posture['torso_lean'])
                    vert_ratio = posture['vertical_distance'] / self.good_posture['vertical_distance']
                    lean_diff = self.good_posture['lean_depth'] - posture['lean_depth']

                    # Per-check thresholds
                    c = self.checks
                    spine_thresh = c['spine']['threshold']
                    lean_thresh = c['head_forward']['threshold'] / 250

                    # Debug
                    print(f"Shoulder: {shoulder_diff:.1f}° | Neck: {neck_diff:.1f}° | Torso: {torso_diff:.1f}° | Vert: {vert_ratio:.2f} | Lean: {lean_diff:.3f}")

                    # Collect specific issues (only for enabled checks)
                    issues = []
                    if c['shoulders']['enabled'] and shoulder_diff > c['shoulders']['threshold']:
                        issues.append("Shoulders tilted")
                    if c['neck']['enabled'] and neck_diff > c['neck']['threshold']:
                        issues.append("Neck angled")
                    if c['head_forward']['enabled'] and lean_diff > lean_thresh:
                        issues.append("Too close to screen")
                    if c['torso']['enabled'] and torso_diff > c['torso']['threshold']:
                        issues.append("Torso tilting")
                    if c['spine']['enabled'] and vert_ratio < (1 - spine_thresh / 100):
                        issues.append("Slouching")

                    if issues:
                        self.posture_issues = issues
                        self.bad_posture_counter += 1
                        print(f"Bad posture ({', '.join(issues)}) counter: {self.bad_posture_counter}")
                        if self.bad_posture_counter > self.alert_delay * 30:  # ~30fps
                            self.show_alert()
                            self.update_status(f"⚠️ {' | '.join(issues)}", "red")
                    else:
                        self.bad_posture_counter = 0
                        self.posture_issues = []
                        self.update_status("✓ Good posture! Keep it up 💚", "lightgreen")
        else:
            self.update_status("👤 No person detected", "white")
        
        # Display frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (480, 360))  # Smaller video feed
        
        # Convert to PhotoImage
        from PIL import Image, ImageTk
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        
        if self.video_label:
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        
        # Schedule next frame
        if self.root:
            self.root.after(33, self.process_frame)  # ~30 FPS
    
    def create_gui(self):
        """Create the main GUI window"""
        self.root = tk.Tk()
        self.root.title("Posture Guardian 🌿")
        self.root.geometry("700x700")
        self.root.configure(bg='#667eea')
        self.root.resizable(True, True)

        # Scrollable container
        canvas = tk.Canvas(self.root, bg='#667eea', highlightthickness=0)
        scrollbar = tk.Scrollbar(self.root, orient='vertical', command=canvas.yview)
        self.scroll_frame = tk.Frame(canvas, bg='#667eea')

        self.scroll_frame.bind(
            '<Configure>',
            lambda e: canvas.configure(scrollregion=canvas.bbox('all'))
        )
        canvas_window = canvas.create_window((0, 0), window=self.scroll_frame, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)

        # Keep scroll_frame width synced with canvas
        def _on_canvas_configure(event):
            canvas.itemconfig(canvas_window, width=event.width)
        canvas.bind('<Configure>', _on_canvas_configure)

        # Mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')
        canvas.bind_all('<MouseWheel>', _on_mousewheel)

        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        # Title
        title = tk.Label(
            self.scroll_frame,
            text="Posture Guardian 🌿",
            font=('Helvetica', 28, 'bold'),
            bg='#667eea',
            fg='white'
        )
        title.pack(pady=20)

        # Video frame
        video_frame = tk.Frame(self.scroll_frame, bg='black')
        video_frame.pack(pady=10)

        self.video_label = tk.Label(video_frame, bg='black')
        self.video_label.pack()

        # Status
        self.status_label = tk.Label(
            self.scroll_frame,
            text="🎯 Click 'Calibrate' to start",
            font=('Helvetica', 14),
            bg='#667eea',
            fg='white',
            pady=15
        )
        self.status_label.pack()

        # Controls
        controls_frame = tk.Frame(self.scroll_frame, bg='#667eea')
        controls_frame.pack(pady=10)
        
        calibrate_btn = tk.Button(
            controls_frame,
            text="Calibrate Good Posture",
            command=self.calibrate_posture,
            font=('Helvetica', 12, 'bold'),
            bg='white',
            fg='#667eea',
            padx=20,
            pady=10,
            relief='raised',
            cursor='hand2'
        )
        calibrate_btn.pack()
        
        # Settings
        settings_frame = tk.LabelFrame(
            self.scroll_frame,
            text="Settings",
            font=('Helvetica', 12, 'bold'),
            bg='#667eea',
            fg='white',
            padx=20,
            pady=15
        )
        settings_frame.pack(pady=15, padx=20, fill='x')

        # Helper to build a +/- row
        def make_setting_row(parent, label_text, value_text, dec_cmd, inc_cmd, hint=None):
            row = tk.Frame(parent, bg='#667eea')
            row.pack(fill='x', pady=6)
            tk.Label(row, text=label_text, bg='#667eea', fg='white',
                     font=('Helvetica', 11)).pack(side='left')
            val_label = tk.Label(row, text=value_text, bg='#667eea', fg='white',
                                 font=('Helvetica', 11, 'bold'), width=5)
            val_label.pack(side='left', padx=10)
            btn_frame = tk.Frame(row, bg='#667eea')
            btn_frame.pack(side='left')
            tk.Button(btn_frame, text="-", command=dec_cmd, font=('Helvetica', 12, 'bold'),
                      bg='white', fg='#667eea', width=2, cursor='hand2').pack(side='left', padx=2)
            tk.Button(btn_frame, text="+", command=inc_cmd, font=('Helvetica', 12, 'bold'),
                      bg='white', fg='#667eea', width=2, cursor='hand2').pack(side='left', padx=2)
            if hint:
                tk.Label(row, text=hint, bg='#667eea', fg='#c0c8f0',
                         font=('Helvetica', 9)).pack(side='left', padx=8)
            return val_label, row

        # Notification type
        type_frame = tk.Frame(settings_frame, bg='#667eea')
        type_frame.pack(fill='x', pady=6)
        tk.Label(type_frame, text="Notification:", bg='#667eea', fg='white',
                 font=('Helvetica', 11)).pack(side='left')
        self.alert_type_label = tk.Label(
            type_frame, text=self.alert_type.capitalize(), bg='#667eea', fg='white',
            font=('Helvetica', 11, 'bold'), width=7)
        self.alert_type_label.pack(side='left', padx=10)
        tk.Button(type_frame, text="Switch", command=self.toggle_alert_type,
                  font=('Helvetica', 10, 'bold'), bg='white', fg='#667eea',
                  cursor='hand2').pack(side='left', padx=2)
        tk.Label(type_frame, text="(sound / overlay popup)", bg='#667eea',
                 fg='#c0c8f0', font=('Helvetica', 9)).pack(side='left', padx=8)

        # Popup duration (only visible when popup is selected)
        self.dur_value, self.dur_row = make_setting_row(
            settings_frame, "Popup Duration:", f"{self.alert_duration}s",
            self.decrease_duration, self.increase_duration, "(how long popup stays)")
        self._update_dur_visibility()

        # Delay before alert fires
        self.delay_value, _ = make_setting_row(
            settings_frame, "Delay:", f"{self.alert_delay}s",
            self.decrease_delay, self.increase_delay, "(seconds before alert fires)")

        # Posture checks with per-check thresholds
        checks_frame = tk.LabelFrame(
            self.scroll_frame,
            text="Posture Checks  (lower = stricter)",
            font=('Helvetica', 12, 'bold'),
            bg='#667eea',
            fg='white',
            padx=20,
            pady=10
        )
        checks_frame.pack(pady=10, padx=20, fill='x')

        self.check_vars = {}
        self.threshold_labels = {}
        check_info = {
            'shoulders':    ('Uneven shoulders',      'One shoulder higher than the other'),
            'neck':         ('Neck bent',             'Head tilting forward or sideways'),
            'head_forward': ('Too close to screen',   'Body leaning toward the monitor'),
            'torso':        ('Leaning to one side',   'Upper body not centered'),
            'spine':        ('Sinking down',          'Spine curving, losing height'),
        }
        for key, (label, desc) in check_info.items():
            # Checkbox + description row
            row = tk.Frame(checks_frame, bg='#667eea')
            row.pack(anchor='w', fill='x', pady=4)

            var = tk.BooleanVar(value=self.checks[key]['enabled'])
            self.check_vars[key] = var
            tk.Checkbutton(
                row, text=label, variable=var,
                command=lambda k=key: self.toggle_check(k),
                bg='#667eea', fg='white', selectcolor='#4a5db8',
                activebackground='#667eea', activeforeground='white',
                font=('Helvetica', 11)
            ).pack(side='left')

            # Threshold +/- buttons (packed right-to-left, so + first then label then -)
            tk.Button(
                row, text="+", font=('Helvetica', 12, 'bold'),
                command=lambda k=key: self.adjust_threshold(k, 1),
                bg='white', fg='#667eea', width=2, cursor='hand2'
            ).pack(side='right', padx=2)

            thresh_label = tk.Label(
                row, text=str(self.checks[key]['threshold']),
                bg='#667eea', fg='white', font=('Helvetica', 11, 'bold'), width=3
            )
            thresh_label.pack(side='right')
            self.threshold_labels[key] = thresh_label

            tk.Button(
                row, text="-", font=('Helvetica', 12, 'bold'),
                command=lambda k=key: self.adjust_threshold(k, -1),
                bg='white', fg='#667eea', width=2, cursor='hand2'
            ).pack(side='right', padx=2)

            # Description on next line
            desc_row = tk.Frame(checks_frame, bg='#667eea')
            desc_row.pack(anchor='w', fill='x')
            tk.Label(desc_row, text=desc, bg='#667eea', fg='#c0c8f0',
                     font=('Helvetica', 10)).pack(side='left', padx=(24, 0))

        # Reflect loaded settings
        self.dur_value.config(text=f"{self.alert_duration}s")
        self.delay_value.config(text=f"{self.alert_delay}s")
        if self.needs_recalibration:
            self.status_label.config(
                text="⚠️ Calibration outdated — please recalibrate", fg='yellow'
            )
        elif self.good_posture:
            self.status_label.config(
                text="✓ Calibration loaded! Monitoring...", fg='lightgreen'
            )

        # Start camera
        self.cap = cv2.VideoCapture(0)
        self.process_frame()

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.root.mainloop()
    
    def increase_duration(self):
        if self.alert_duration < 10:
            self.alert_duration += 1
            self.dur_value.config(text=f"{self.alert_duration}s")
            self.save_calibration()

    def decrease_duration(self):
        if self.alert_duration > 1:
            self.alert_duration -= 1
            self.dur_value.config(text=f"{self.alert_duration}s")
            self.save_calibration()

    def increase_delay(self):
        if self.alert_delay < 30:
            self.alert_delay += 1
            self.delay_value.config(text=f"{self.alert_delay}s")
            self.save_calibration()

    def decrease_delay(self):
        if self.alert_delay > 1:
            self.alert_delay -= 1
            self.delay_value.config(text=f"{self.alert_delay}s")
            self.save_calibration()

    def toggle_alert_type(self):
        """Switch between sound and popup alert types"""
        self.alert_type = 'popup' if self.alert_type == 'sound' else 'sound'
        self.alert_type_label.config(text=self.alert_type.capitalize())
        self._update_dur_visibility()
        self.save_calibration()

    def _update_dur_visibility(self):
        """Show/hide popup duration row based on alert type"""
        for child in self.dur_row.winfo_children():
            if self.alert_type == 'popup':
                child.configure(state='normal')
            else:
                try:
                    child.configure(state='disabled')
                except:
                    pass
        # Dim the row when not applicable
        fg = 'white' if self.alert_type == 'popup' else '#667eea'
        for child in self.dur_row.winfo_children():
            try:
                child.configure(fg=fg)
            except:
                pass

    def toggle_check(self, key):
        """Toggle a posture check on/off"""
        self.checks[key]['enabled'] = self.check_vars[key].get()
        self.save_calibration()

    def adjust_threshold(self, key, delta):
        """Adjust a per-check threshold by delta"""
        new_val = self.checks[key]['threshold'] + delta
        if 1 <= new_val <= 20:
            self.checks[key]['threshold'] = new_val
            self.threshold_labels[key].config(text=str(new_val))
            self.save_calibration()

    def on_closing(self):
        """Clean up on window close"""
        self.monitoring = False
        if self.cap:
            self.cap.release()
        if self.root:
            self.root.destroy()
    
    def run(self):
        """Start the application"""
        self.create_gui()

if __name__ == "__main__":
    app = PostureGuardian()
    app.run()
