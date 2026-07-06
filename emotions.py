import numpy as np
import cv2
import pyDatalog
import json
from pyDatalog import pyDatalog
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tkinter import *
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import os
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime
from tkinter import ttk, messagebox


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
cv2.ocl.setUseOpenCL(False)

# Emotion dictionary
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful",
                3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Initialize pyDatalog
pyDatalog.create_terms('emotion, X, Y')
for i, emo in emotion_dict.items():
    pyDatalog.assert_fact('emotion', i, emo)

# Create model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))
model.load_weights('model.h5')

# Load face cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


class EmotionDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Detection System")
        self.root.geometry("1000x800")
        self.open_windows = []

        # Initialize style and theme
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Load theme with fallback
        self.current_theme = self.load_theme_from_config()
        self.load_theme()
        self.setup_main_ui()

        # Emotion tracking variables
        self.emotion_history = deque(maxlen=60)
        self.intensity_var = DoubleVar(value=0)
        self.thresholds = {'Angry': 0.7, 'Happy': 0.8}
        self.alert_cooldown = False

    def load_theme_from_config(self):
        """
        Load theme from theme_config.json with fallback to 'Blue'
        """
        default_theme = 'Blue'
        try:
            with open('theme_config.json', 'r') as f:
                config = json.load(f)
                theme = config.get('theme', default_theme)
                # Validate theme exists
                valid_themes = ['Pink', 'Gray', 'Yellow', 'Blue']
                if theme in valid_themes:
                    return theme
                else:
                    print(f"⚠️ Invalid theme '{theme}' in config, using '{default_theme}'")
                    return default_theme
        except FileNotFoundError:
            print("ℹ️ theme_config.json not found, using default 'Blue' theme")
            return default_theme
        except json.JSONDecodeError:
            print("⚠️ theme_config.json is corrupted, using default 'Blue' theme")
            return default_theme
        except Exception as e:
            print(f"⚠️ Error loading theme config: {e}, using default 'Blue'")
            return default_theme

    def setup_main_ui(self):
        # Main container
        self.main_frame = Frame(self.root, bg=self.colors['background'])
        self.main_frame.pack(expand=True, fill=BOTH)

        # Header (modified to store reference)
        self.header_frame = Frame(self.main_frame, bg=self.colors['primary'])
        self.header_frame.pack(fill=X, pady=20)
        self.header_label = Label(
            self.header_frame, 
            text="Emotion Detection AI",
            font=('Arial', 24, 'bold'), 
            fg='black',
            bg=self.colors['primary']
        )
        self.header_label.pack(pady=20)
        
        # Button container
        btn_container = Frame(self.main_frame, bg=self.colors['background'])
        btn_container.pack(pady=20)

        # Action buttons
        self.create_button(btn_container, "🎥 Real-time Detection", self.start_realtime)
        self.create_button(btn_container, "📁 Upload Image", self.upload_image)
        self.create_button(btn_container, "⚙ Theme Settings", self.open_settings)

        # Video capture reference
        self.cap = None
        self.realtime_window = None

    def create_button(self, parent, text, command):
        btn = ttk.Button(parent, text=text, command=command, style='Custom.TButton')
        btn.pack(pady=10, padx=20, fill=X)

    def start_realtime(self):
        if self.realtime_window is None:
            self.realtime_window = Toplevel(self.root)
            self.open_windows.append(self.realtime_window)
            self.realtime_window.title("Real-time Detection")
            self.realtime_window.geometry("1200x900")
            self.apply_theme_to_window(self.realtime_window)

            # Main container using grid
            main_container = Frame(self.realtime_window, bg=self.colors['background'])
            main_container.pack(fill=BOTH, expand=True)

            # Camera and button container
            top_container = Frame(main_container, bg=self.colors['background'])
            top_container.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)

            # Camera display
            video_frame = Frame(top_container, bg=self.colors['background'])
            video_frame.grid(row=0, column=0, sticky="nsew")
            self.video_label = Label(video_frame, bg=self.colors['background'])
            self.video_label.pack()

            # Back button frame (right side)
            button_frame = Frame(top_container, bg=self.colors['background'])
            button_frame.grid(row=0, column=1, sticky="ne", padx=20)

            # Back to Home button
            back_btn = ttk.Button(
                button_frame, 
                text="⬅ Back to Home",
                command=self.stop_realtime,
                style='TButton'
            )
            back_btn.pack(pady=10, anchor='ne')

            # Emotion timeline graph (below camera)
            graph_frame = Frame(main_container, bg=self.colors['background'])
            graph_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=20)
            self.create_emotion_graph(graph_frame)

            # Configure grid weights
            main_container.grid_rowconfigure(0, weight=1)
            main_container.grid_rowconfigure(1, weight=3)
            main_container.grid_columnconfigure(0, weight=1)
            top_container.grid_columnconfigure(0, weight=4)  # 80% space for camera
            top_container.grid_columnconfigure(1, weight=1)  # 20% space for button

            # Start video capture
            self.cap = cv2.VideoCapture(0)
            self.update_realtime_frame()

    def create_emotion_graph(self, parent):
        self.fig = plt.Figure(figsize=(10, 4.5), dpi=90)
        self.ax = self.fig.add_subplot(111)
        self.graph_canvas = FigureCanvasTkAgg(self.fig, parent)
        self.graph_canvas.get_tk_widget().configure(
            bg=self.colors['background'], 
            height=350
        )
        self.graph_canvas.get_tk_widget().pack(fill=BOTH, expand=True)

    def update_video_display(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img.thumbnail((640, 480))
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    def update_emotion_graph(self):
        self.ax.clear()
        times = [entry[0] for entry in self.emotion_history]
        emotions = [entry[1] for entry in self.emotion_history]

        # Create numeric y-values with spacing
        y = [list(emotion_dict.values()).index(e) * 1.2 for e in emotions]

        self.ax.plot(times, y, marker='o', linestyle='-', color='#ff69b4', markersize=8)

        # Configure axes
        self.ax.set_yticks([i * 1.2 for i in range(len(emotion_dict))])
        self.ax.set_yticklabels(emotion_dict.values(), fontsize=10)
        self.ax.set_title("Emotion Timeline", fontsize=12, color=self.colors['text'])
        self.ax.grid(alpha=0.3)

        # Format x-axis
        self.ax.xaxis.set_major_locator(plt.MaxNLocator(8))
        plt.setp(self.ax.get_xticklabels(), rotation=30, ha='right', fontsize=8)

        # Styling
        self.ax.set_facecolor(self.colors['background'])
        self.fig.patch.set_facecolor(self.colors['background'])
        self.fig.tight_layout(pad=2.5)

        self.graph_canvas.draw()

    def update_realtime_frame(self):
        ret, frame = self.cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            max_confidence = 0
            for i, (x, y, w, h) in enumerate(faces):
                # Face processing
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 105, 180), 2)
                roi_gray = gray[y:y+h, x:x+w]

                # Normalize and prepare for model
                resized_img = cv2.resize(roi_gray, (48, 48)) / 255.0
                cropped_img = np.expand_dims(np.expand_dims(resized_img, -1), 0)

                prediction = model.predict(cropped_img)
                maxindex = int(np.argmax(prediction))
                confidence = np.max(prediction)
                emotion = (
                    pyDatalog.ask(f'emotion({maxindex}, X)').answers[0][0]
                    if pyDatalog.ask(f'emotion({maxindex}, X)')
                    else "Unknown"
                )

                # Update intensity
                if confidence > max_confidence:
                    max_confidence = confidence
                    self.intensity_var.set(confidence * 100)
                    self.update_intensity_color(confidence)

                # Multi-face labels
                label = f"Face {i+1}: {emotion} ({confidence*100:.1f}%)"
                cv2.putText(
                    frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                )

                # Store history
                self.emotion_history.append((datetime.now(), emotion, confidence))

            # Update displays
            self.update_video_display(frame)
            if len(faces) > 0:
                self.update_emotion_graph()

            self.video_label.after(10, self.update_realtime_frame)
        else:
            self.stop_realtime()

    def update_intensity_color(self, confidence):
        color = '#ffb3d9' if confidence < 0.5 else '#ff69b4' if confidence < 0.8 else '#ff1493'
        self.style.configure('Intensity.Horizontal.TProgressbar', background=color)

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            # Store processed image for download
            self.processed_image = cv2.imread(file_path)
            gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            # Process image with emotions
            for (x, y, w, h) in faces:
                # Draw face rectangle
                cv2.rectangle(self.processed_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Get emotion prediction
                roi_gray = gray[y:y+h, x:x+w]
                resized_img = cv2.resize(roi_gray, (48, 48)) / 255.0
                cropped_img = np.expand_dims(np.expand_dims(resized_img, -1), 0)
                prediction = model.predict(cropped_img)
                maxindex = int(np.argmax(prediction))

                result = pyDatalog.ask(f'emotion({maxindex}, X)')
                emotion = list(result.answers)[0][0] if result else "Unknown"

                # Calculate text position
                label_y = y + h + 20
                text_size = cv2.getTextSize(emotion, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                text_x = x + (w - text_size[0]) // 2

                # Ensure text stays within image boundaries
                if label_y + text_size[1] > self.processed_image.shape[0]:
                    label_y = y - 10

                # Draw background rectangle
                cv2.rectangle(
                    self.processed_image,
                    (text_x - 5, label_y - text_size[1] - 5),
                    (text_x + text_size[0] + 5, label_y + 5),
                    (0, 255, 0), -1
                )

                # Put emotion text
                cv2.putText(
                    self.processed_image, emotion, (text_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                )

            # Create result window
            result_window = Toplevel(self.root)
            result_window.title("Image Analysis Result")
            self.open_windows.append(result_window)
            self.apply_theme_to_window(result_window)

            # Display image
            display_img = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB)
            display_img = Image.fromarray(display_img)
            display_img.thumbnail((1000, 800))
            img_tk = ImageTk.PhotoImage(display_img)

            image_label = Label(result_window, image=img_tk, bg=self.colors['background'])
            image_label.image = img_tk
            image_label.pack(padx=20, pady=20)

            # Control buttons frame
            btn_frame = Frame(result_window, bg=self.colors['background'])
            btn_frame.pack(pady=10)

            # Download button
            download_btn = ttk.Button(
                btn_frame, 
                text="💾 Download Image",
                command=lambda: self.download_image()
            )
            download_btn.pack(side=LEFT, padx=5)

            # Back button
            back_btn = ttk.Button(
                btn_frame, 
                text="⬅ Back to Main",
                command=result_window.destroy
            )
            back_btn.pack(side=LEFT, padx=5)

    def download_image(self):
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".jpg",
                filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")]
            )
            if file_path:
                cv2.imwrite(file_path, self.processed_image)
                messagebox.showinfo("Success", "Image downloaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save image: {str(e)}")

    # Theme management functions
    def load_theme(self):
        themes = {
            'Pink': {
                'background': '#ffcccb',
                'primary': '#ff69b4',
                'secondary': '#ff1493',
                'text': '#000000',
                'hover': '#ff007f'
            },
            'Gray': {
                'background': '#f0f0f0',
                'primary': '#d0d0d0',
                'secondary': '#696969',
                'text': '#000000',
                'hover': '#696969'
            },
            'Yellow': {
                'background': '#fff9c4',
                'primary': '#ffd54f',
                'secondary': '#ffb300',
                'text': '#000000',
                'hover': '#ffa000'
            },
            'Blue': {
                'background': '#e3f2fd',
                'primary': '#42a5f5',
                'secondary': '#1976d2',
                'text': '#000000',
                'hover': '#1565c0'
            }
        }
        self.colors = themes[self.current_theme]
        self.configure_styles()

    def configure_styles(self):
        self.style.configure(
            'TButton',
            background=self.colors['secondary'],
            foreground=self.colors['text'],
            padding=10,
            font=('Arial', 12)
        )
        self.style.map(
            'TButton',
            background=[('active', self.colors['hover'])]
        )

    def apply_theme_to_window(self, window):
        window.configure(bg=self.colors['background'])
        for widget in window.winfo_children():
            try:
                if isinstance(widget, (Label, Frame)):
                    widget.configure(bg=self.colors['primary'], fg=self.colors['text'])
                elif isinstance(widget, ttk.Button):
                    widget.configure(style='TButton')
            except:
                pass

    def open_settings(self):
        # Create settings window
        self.settings_window = Toplevel(self.root)
        self.settings_window.title("Theme Settings")
        self.settings_window.geometry("300x200")
        self.open_windows.append(self.settings_window)
        self.apply_theme_to_window(self.settings_window)

        # Theme selection label
        Label(
            self.settings_window, 
            text="Select Theme:",
            font=('Arial', 12), 
            bg=self.colors['primary'],
            fg=self.colors['text']
        ).pack(pady=10)

        # Theme dropdown
        self.theme_var = StringVar(value=self.current_theme)
        theme_menu = ttk.Combobox(
            self.settings_window,
            textvariable=self.theme_var,
            values=['Pink', 'Gray', 'Yellow', 'Blue'],
            state='readonly'
        )
        theme_menu.pack(pady=5)

        # Save button
        save_btn = ttk.Button(
            self.settings_window,
            text="Apply Theme",
            command=self.save_theme_settings
        )
        save_btn.pack(pady=15)

    def save_theme_settings(self):
        new_theme = self.theme_var.get()
        if new_theme != self.current_theme:
            self.current_theme = new_theme
            # Save to config file
            with open('theme_config.json', 'w') as f:
                json.dump({'theme': new_theme}, f)
            # Apply changes
            self.load_theme()
            self.apply_theme_to_window(self.root)
            self.update_main_window_colors()
            # Update all open windows
            for window in self.open_windows:
                self.apply_theme_to_window(window)
        self.settings_window.destroy()

    def update_main_window_colors(self):
        # Update main window background
        self.root.configure(bg=self.colors['background'])

        # Update header frame and label
        self.header_frame.configure(bg=self.colors['primary'])
        self.header_label.configure(
            bg=self.colors['primary'],
            fg=self.colors['text']
        )
        # Update main frame and its children
        self.main_frame.configure(bg=self.colors['background'])
        for widget in self.main_frame.winfo_children():
            try:
                if isinstance(widget, Frame):
                    widget.configure(bg=self.colors['primary'])
                elif isinstance(widget, Label):
                    widget.configure(bg=self.colors['primary'], fg=self.colors['text'])
                elif isinstance(widget, ttk.Button):
                    widget.configure(style='TButton')
            except:
                pass

    def stop_realtime(self):
        if self.cap:
            self.cap.release()
        if self.realtime_window:
            self.realtime_window.destroy()
            self.realtime_window = None


if __name__ == "__main__":
    root = Tk()
    app = EmotionDetectionApp(root)
    root.mainloop()