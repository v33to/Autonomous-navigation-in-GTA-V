import cv2
import os
import time
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk

"""
Speedometer digit recognition tool that uses an SVM model to extract speed values from the interface.
"""

path = os.getcwd()
path += '/SpeedAcquisition/'

try:
    svm = cv2.ml.SVM_load(path + 'svm.yml')
except cv2.error:
    print("Error loading SVM model. Check path and file.")
    exit()

def get_speed(image):
    """
    Extract and process speedometer digits to get speed value
    """
    if image.shape[:2] != (863, 1152):
        return -2
        
    y_positions = [0, -50, -80]
    found_speed = False
    
    for i in y_positions:
        y1 = 757 + i
        y2 = 793 + i
        speedometer = image[y1:y2, 1015:1093]
        speedometer = cv2.cvtColor(speedometer, cv2.COLOR_BGR2GRAY)
        _, speedometer = cv2.threshold(speedometer, 254, 255, cv2.THRESH_BINARY)
        
        coordinates = [(0, 25), (26, 51), (52, 77)]
        digit_images = [speedometer[0:35, j[0]:j[1]] for j in coordinates]
        speed = 0
        n = 100
        
        for digit_img in digit_images:
            bit_array = digit_img.ravel() >= 250
            _, result = svm.predict(np.array([bit_array.astype(int)], dtype=np.float32))
            digit = result[0][0]
            
            if digit not in (10, 11):
                speed += digit * n
                found_speed = True
            n //= 10
        
        if found_speed:
            break
    
    if found_speed and speed < 260:
        return speed
    else:
        return -1
    
_last_valid_speed = 0.0

def get_speed_from_file():
    """
    Read the current speed from GTA V via the speed file.
    Returns last known good value if current read fails.
    """
    global _last_valid_speed
    
    speed_file = "C:/Games/Grand Theft Auto V/scripts/speed.txt"
    
    try:
        with open(speed_file, 'r') as f:
            content = f.read().strip()
            if content:
                speed = float(content)
                _last_valid_speed = speed
                return speed
    except:
        pass
    
    return _last_valid_speed

def image_processing(path_image):
    """
    Process the selected image and update the UI
    """
    try:
        image = cv2.imread(path_image)
        if image is None:
            raise cv2.error(f"Error loading image, not found or corrupted.")
    except (cv2.error, ValueError) as e:
        print(e)
        speed_label.configure(text="Error loading image.")
        return
        
    start = time.time()
    speed = get_speed(image)
    prediction_time = time.time() - start
    
    if speed == -2:
        error = "Incorrect image dimensions."
        print(error)
        speed_label.configure(text=error)
        prediction_time_label.configure(text="")
        image_label.image = None
        image_label.configure(image='')
        return
    
    if speed == -1:
        speed_label.configure(text="Speed was not recognized.")
    else:
        speed_label.configure(text=f"Speed: {speed}")
        
    prediction_time_label.configure(text=f"Prediction took {prediction_time:.7f} seconds")
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image = ImageTk.PhotoImage(image)
    image_label.configure(image=image)
    image_label.image = image

def select_image():
    """
    Open file dialog and process selected image
    """
    path_image = filedialog.askopenfilename(
        title="Select image",
        filetypes=[("Png, jpg, bmp images", "*.png;*.jpg;*.bmp")]
    )
    if path_image:
        image_processing(path_image)

if __name__ == "__main__":
    """
    Initialize and run the GUI application for speed recognition testing with file selection menu.
    """
    font = ('Helvetica', 16)
    window = tk.Tk()
    window.geometry("+400+0")
    window.title("Speed Recognition")
    
    menu = tk.Menu(window)
    window.config(menu=menu)
    file_menu = tk.Menu(menu, tearoff=0)
    menu.add_cascade(label="File", menu=file_menu)
    file_menu.add_command(label="Load image", command=select_image)
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=window.quit)
    
    frame = ttk.Frame(window, padding=10)
    frame.grid(row=0, column=0, sticky='nsew')
    image = ttk.Frame(frame)
    image.grid(row=0, column=0, padx=10, pady=10)
    image_label = tk.Label(image)
    image_label.grid(row=0, column=0)
    
    message = ttk.Frame(frame)
    message.grid(row=1, column=0, padx=10, pady=10)
    speed_label = ttk.Label(message, text="", font=font, width=30, anchor="w")
    speed_label.grid(row=0, column=0, padx=5)
    prediction_time_label = ttk.Label(message, text="", font=font)
    prediction_time_label.grid(row=0, column=1, padx=5)
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)
    frame.mainloop()