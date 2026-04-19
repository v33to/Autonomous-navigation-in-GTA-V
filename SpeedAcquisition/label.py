import os
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import cv2
from pathlib import Path
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import shutil

"""
Digit classification tool that uses K-means clustering to group similar images,
then uses a GUI for manual verification and labeling into organized directories.
"""

path = os.getcwd()
path += '/SpeedAcquisition/Digits'

class DigitLabeler:
    def __init__(self, data_path):
        """
        Initialize the digit labeler with data path and create necessary directories for labeled data.
        """
        self.data_path = Path(data_path)
        self.labeled_path = self.data_path / 'labeled'
        self.labeled_path.mkdir(exist_ok=True)
        
        self.classes = [str(i) for i in range(10)] + ['black', 'progress_bar']
        for cls in self.classes:
            (self.labeled_path / cls).mkdir(exist_ok=True)
        
        self.images = []
        self.image_paths = []
        self.current_cluster = 0
        self.clustered_images = {}
        
    def cluster_images(self, n_clusters=12):
        """
        Load images and perform K-means clustering to group similar digit images together.
        """
        self.images = []
        self.image_paths = []
        for img_path in self.data_path.glob('digit_*.png'):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                self.images.append(img.flatten())
                self.image_paths.append(img_path)
        
        X = np.array(self.images)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        self.clustered_images = {}
        for img_path, cluster in zip(self.image_paths, clusters):
            self.clustered_images.setdefault(cluster, []).append(img_path)
            
        return self.clustered_images
    
    def create_labeling_ui(self):
        """
        Create UI for manual verification and labeling of clustered images
        """
        self.root = tk.Tk()
        self.root.geometry("+850+250")
        self.root.title("Digit Labeler")
        
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.image_frame = ttk.Frame(main_frame)
        self.image_frame.grid(row=0, column=0, columnspan=3)
        controls_frame = ttk.Frame(main_frame)
        controls_frame.grid(row=1, column=0, columnspan=3, pady=10)
        
        ttk.Label(controls_frame, text="Current cluster:").grid(row=0, column=0)
        self.cluster_label = ttk.Label(controls_frame, text="0")
        self.cluster_label.grid(row=0, column=1)
        
        ttk.Label(controls_frame, text="Assign as:").grid(row=1, column=0)
        self.class_var = tk.StringVar()
        class_combo = ttk.Combobox(controls_frame, textvariable=self.class_var)
        class_combo['values'] = self.classes
        class_combo.grid(row=1, column=1)
        ttk.Button(controls_frame, text="Assign & Next", 
                  command=self.assign_and_next).grid(row=1, column=2)
        
        shortcuts_text = """
        Shortcuts:
        0-9: Assign as digit
        b: Assign as black
        p: Assign as progress bar
        n: Next cluster
        z: Previous cluster
        """
        ttk.Label(main_frame, text=shortcuts_text).grid(row=2, column=0, columnspan=3)
        self.root.bind('<Key>', self.handle_keypress)
        self.show_current_cluster()
        self.root.mainloop()
    
    def show_current_cluster(self):
        """
        Display up to 25 images from the current cluster in a 5x5 grid layout.
        """
        for widget in self.image_frame.winfo_children():
            widget.destroy()
        
        self.cluster_label.config(text=str(self.current_cluster))
        
        if self.current_cluster in self.clustered_images:
            paths = self.clustered_images[self.current_cluster][:25]
            for i, path in enumerate(paths):
                img = Image.open(path)
                img = img.resize((50, 50))
                photo = ImageTk.PhotoImage(img)
                label = ttk.Label(self.image_frame, image=photo)
                label.image = photo
                label.grid(row=i//5, column=i%5, padx=2, pady=2)
    
    def assign_and_next(self):
        """
        Copy all images from current cluster to the selected class directory and move to next cluster.
        """
        if not self.class_var.get() or self.current_cluster not in self.clustered_images:
            return
            
        target_dir = self.labeled_path / self.class_var.get()
        for path in self.clustered_images[self.current_cluster]:
            shutil.copy(path, target_dir / path.name)
            
        self.current_cluster += 1
        if self.current_cluster >= len(self.clustered_images):
            self.root.quit()
        else:
            self.show_current_cluster()
    
    def handle_keypress(self, event):
        """
        Handle keyboard shortcuts for quick labeling and navigation between clusters.
        """
        if event.char in [str(i) for i in range(10)]:
            self.class_var.set(event.char)
            self.assign_and_next()
        elif event.char == 'b':
            self.class_var.set('black')
            self.assign_and_next()
        elif event.char == 'p':
            self.class_var.set('progress_bar')
            self.assign_and_next()
        elif event.char == 'n':
            self.current_cluster = min(self.current_cluster + 1, len(self.clustered_images) - 1)
            self.show_current_cluster()
        elif event.char == 'z':
            self.current_cluster = max(0, self.current_cluster - 1)
            self.show_current_cluster()

def analyze_dataset(labeled_path):
    """
    Analyze and display statistics of the labeled dataset showing image counts per class.
    """
    labeled_path = Path(labeled_path)
    stats = {}
    for cls_dir in labeled_path.iterdir():
        if cls_dir.is_dir():
            stats[cls_dir.name] = len(list(cls_dir.glob('*.png')))
    
    print("\nDataset Statistics:")
    print("-----------------")
    for cls, count in sorted(stats.items()):
        print(f"{cls}: {count} images")
    
    return stats

if __name__ == "__main__":
    labeler = DigitLabeler(path)
    labeler.cluster_images()
    labeler.create_labeling_ui()
    stats = analyze_dataset(path + '/labeled')