"""
Main Application - GUI với Tkinter
Kéo thả ảnh vào là predict luôn - SIÊU NHANH
"""
import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import numpy as np
import threading
import time


# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import *
from src.inference_engine import get_engine





class ModernButton(tk.Canvas):
    """Custom button với hover effect"""
    
    def __init__(self, parent, text, command, **kwargs):
        super().__init__(parent, **kwargs)
        self.command = command
        self.text = text
        self.is_hover = False
        
        self.config(
            width=kwargs.get('width', 200),
            height=kwargs.get('height', 50),
            bg=UI_CONFIG['bg_color'],
            highlightthickness=0
        )
        
        self.bind('<Button-1>', lambda e: self.command())
        self.bind('<Enter>', self.on_enter)
        self.bind('<Leave>', self.on_leave)
        
        self.draw()
    
    def draw(self):
        self.delete('all')
        color = '#4CAF50' if not self.is_hover else '#45a049'
        
        self.create_rectangle(
            5, 5, self.winfo_reqwidth()-5, self.winfo_reqheight()-5,
            fill=color, outline='', width=0
        )
        
        self.create_text(
            self.winfo_reqwidth()//2, self.winfo_reqheight()//2,
            text=self.text, fill='white', font=('Segoe UI', 12, 'bold')
        )
    
    def on_enter(self, e):
        self.is_hover = True
        self.draw()
    
    def on_leave(self, e):
        self.is_hover = False
        self.draw()


class RecognitionApp:
    """Main Application"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Recognition - MNIST & Shape Detector")
        self.root.geometry("1000x700")
        self.root.configure(bg=UI_CONFIG['bg_color'])
        
        # Engine
        print("Loading inference engine...")
        self.engine = get_engine()
        print("Engine loaded!")
        
        # Variables
        self.current_image = None
        self.current_image_path = None
        self.is_processing = False
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup UI components"""
        
        # Title
        title_frame = tk.Frame(self.root, bg=UI_CONFIG['bg_color'])
        title_frame.pack(pady=20)
        
        title = tk.Label(
            title_frame,
            text="🤖 Neural Recognition System",
            font=('Segoe UI', 28, 'bold'),
            bg=UI_CONFIG['bg_color'],
            fg='#4CAF50'
        )
        title.pack()
        
        subtitle = tk.Label(
            title_frame,
            text="Drag & Drop image or click Browse - Ultra Fast Inference",
            font=('Segoe UI', 12),
            bg=UI_CONFIG['bg_color'],
            fg='#888888'
        )
        subtitle.pack()
        
        # Main container
        main_container = tk.Frame(self.root, bg=UI_CONFIG['bg_color'])
        main_container.pack(expand=True, fill='both', padx=30, pady=10)
        
        # Left panel - Image preview
        left_panel = tk.Frame(main_container, bg='#1e1e1e', relief='solid', bd=2)
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 15))
        
        preview_label = tk.Label(
            left_panel,
            text="Image Preview",
            font=('Segoe UI', 14, 'bold'),
            bg='#1e1e1e',
            fg='white'
        )
        preview_label.pack(pady=10)
        
        # Drop zone
        self.drop_zone = tk.Label(
            left_panel,
            text="📁\n\nDrag & Drop Image Here\nor\nClick Browse Button",
            font=('Segoe UI', 16),
            bg='#2d2d2d',
            fg='#888888',
            relief='solid',
            bd=2,
            cursor='hand2'
        )
        self.drop_zone.pack(expand=True, fill='both', padx=20, pady=20)
        self.drop_zone.bind('<Button-1>', lambda e: self.browse_image())
        
        # Browse button
        btn_frame = tk.Frame(left_panel, bg='#1e1e1e')
        btn_frame.pack(pady=15)
        
        browse_btn = ModernButton(
            btn_frame,
            text="📂 Browse Image",
            command=self.browse_image,
            width=180,
            height=45
        )
        browse_btn.pack()
        
        # Right panel - Results
        right_panel = tk.Frame(main_container, bg='#1e1e1e', relief='solid', bd=2)
        right_panel.pack(side='right', fill='both', expand=True)
        
        result_label = tk.Label(
            right_panel,
            text="Recognition Result",
            font=('Segoe UI', 14, 'bold'),
            bg='#1e1e1e',
            fg='white'
        )
        result_label.pack(pady=10)
        
        # Result display
        result_container = tk.Frame(right_panel, bg='#2d2d2d')
        result_container.pack(expand=True, fill='both', padx=20, pady=20)
        
        self.result_type = tk.Label(
            result_container,
            text="Type: -",
            font=('Segoe UI', 14),
            bg='#2d2d2d',
            fg='#4CAF50',
            anchor='w'
        )
        self.result_type.pack(fill='x', pady=10, padx=20)
        
        self.result_value = tk.Label(
            result_container,
            text="🔍",
            font=('Segoe UI', 72, 'bold'),
            bg='#2d2d2d',
            fg='white'
        )
        self.result_value.pack(expand=True)
        
        self.result_confidence = tk.Label(
            result_container,
            text="Confidence: -",
            font=('Segoe UI', 16),
            bg='#2d2d2d',
            fg='#888888'
        )
        self.result_confidence.pack(pady=10)
        
        self.inference_time = tk.Label(
            result_container,
            text="Inference Time: -",
            font=('Segoe UI', 12),
            bg='#2d2d2d',
            fg='#666666'
        )
        self.inference_time.pack(pady=5)
        
        # Status bar
        self.status_bar = tk.Label(
            self.root,
            text="Ready • Models loaded",
            font=('Segoe UI', 10),
            bg='#1e1e1e',
            fg='#4CAF50',
            anchor='w',
            padx=10
        )
        self.status_bar.pack(side='bottom', fill='x')
        
        # Enable drag and drop
        self.root.drop_target_register(tk.DND_FILES)
        self.root.dnd_bind('<<Drop>>', self.on_drop)
    
    def on_drop(self, event):
        """Handle drag and drop"""
        files = self.root.tk.splitlist(event.data)
        if files:
            self.load_and_predict(files[0])
    
    def browse_image(self):
        """Browse and select image"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.load_and_predict(file_path)
    
    def load_and_predict(self, file_path):
        """Load image and predict"""
        if self.is_processing:
            return
        
        try:
            self.is_processing = True
            self.status_bar.config(text="Loading image...", fg='#FFA500')
            self.root.update()
            
            # Load image
            image = Image.open(file_path)
            self.current_image = image
            self.current_image_path = file_path
            
            # Display image
            self.display_image(image)
            
            # Predict in separate thread
            thread = threading.Thread(target=self.predict_image, args=(image,))
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            self.status_bar.config(text=f"Error: {str(e)}", fg='#FF0000')
            self.is_processing = False
    
    def display_image(self, image):
        """Display image in preview"""
        # Resize for display
        display_size = (UI_CONFIG['preview_size'][0] - 40, UI_CONFIG['preview_size'][1] - 40)
        
        # Calculate aspect ratio
        img_ratio = image.width / image.height
        display_ratio = display_size[0] / display_size[1]
        
        if img_ratio > display_ratio:
            new_width = display_size[0]
            new_height = int(new_width / img_ratio)
        else:
            new_height = display_size[1]
            new_width = int(new_height * img_ratio)
        
        display_img = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(display_img)
        
        self.drop_zone.config(image=photo, text='')
        self.drop_zone.image = photo
    
    def predict_image(self, image):
        """Predict image in separate thread"""
        try:
            self.status_bar.config(text="Processing...", fg='#FFA500')
            
            # Convert to numpy
            img_array = np.array(image)
            
            # Predict with timing
            start_time = time.perf_counter()
            result = self.engine.predict_auto(img_array)
            inference_time = (time.perf_counter() - start_time) * 1000
            
            # Update UI in main thread
            self.root.after(0, self.update_results, result, inference_time)
            
        except Exception as e:
            self.root.after(0, lambda: self.status_bar.config(
                text=f"Prediction error: {str(e)}", fg='#FF0000'
            ))
        finally:
            self.is_processing = False
    
    def update_results(self, result, inference_time):
        """Update UI with results"""
        # Type
        type_text = "Detected: " + ("Digit" if result['type'] == 'digit' else "Shape")
        self.result_type.config(text=type_text)
        
        # Value
        value_text = str(result['result']).upper()
        self.result_value.config(text=value_text)
        
        # Confidence
        confidence = result['confidence'] * 100
        conf_text = f"Confidence: {confidence:.1f}%"
        self.result_confidence.config(text=conf_text)
        
        # Inference time
        time_text = f"⚡ Inference Time: {inference_time:.1f}ms"
        self.inference_time.config(text=time_text)
        
        # Status
        self.status_bar.config(
            text=f"Ready • Last prediction: {result['result']} ({confidence:.1f}%)",
            fg='#4CAF50'
        )


def main():
    """Main entry point"""
    root = tk.Tk()
    
    # Set icon if available
    try:
        root.iconbitmap('icon.ico')
    except:
        pass
    
    app = RecognitionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()