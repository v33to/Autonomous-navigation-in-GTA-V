import difflib
import win32gui
import win32api
import dxcam
import time
import numpy as np
import cv2
import os
import glob

"""
Script that captures and extracts unique digit images from the speedometer in real-time by taking 
screenshots of the game window, processing the speedometer region and saving individual digit segments.
"""

path = os.getcwd()
path += '/SpeedAcquisition/Digits'
os.makedirs(path, exist_ok=True)
digit_arrays = []

def search(handle, parameters):
    """
    Function that finds window handles by matching titles using fuzzy string comparison.
    """
    text = win32gui.GetWindowText(handle)
    if all(ord(char) <= 127 for char in text) and text.rstrip() == text:
        title = parameters[0]
        percent = difflib.SequenceMatcher(None, text, title).ratio()
        if percent > parameters[1]:
            parameters[1] = percent
            parameters[2] = handle
    return True

def move(title, x = 0, y = 0):
    """
    Finds and repositions a window with the specified title to given coordinates.
    """
    parameters = [title, 0, None]
    win32gui.EnumWindows(search, parameters)
    if parameters[2] is not None:
        coordinates = win32gui.GetWindowRect(parameters[2])
        width = coordinates[2] - coordinates[0]
        height = coordinates[3] - coordinates[1]
        win32gui.MoveWindow(parameters[2], x, y, width, height, True)
    else:
        print(f"'{title}' not found!")

def save_if_unique(img_array, digit_type, counter):
    """
    Checks if a digit image is unique by comparing against existing arrays and saves if new.
    """
    for existing_array in digit_arrays:
        if np.array_equal(img_array, existing_array):
            return False
            
    digit_arrays.append(img_array.copy())
    filename = f'digit_{counter}_{digit_type}.png'
    cv2.imwrite(f"{path}/{filename}", img_array)
    return True

if __name__ == "__main__":
    """
    Main loop that captures speedometer screenshots, extracts digit regions and saves unique images.
    Uses keyboard controls for pause/resume and stopping the capture process (HOME and END).
    """
    move("RAGE Multiplayer")
    roi = (0, 31, 1152, 894)
    camera = dxcam.create(output_idx=0)
    camera.start(region=roi, target_fps=30, video_mode=True)
    counter = 0
    pause = False
    key_pressed = False
    
    if os.path.exists(path):
        for img_path in glob.glob(os.path.join(path, 'digit_*.png')):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            digit_arrays.append(img)
        print(f"Loaded {len(digit_arrays)} existing images")

    for i in list(range(5))[::-1]:
        print(i+1)
        time.sleep(1)

    try:
        while True:
            if(win32api.GetAsyncKeyState(0x24)&0x8001 > 0):
                print(f"\nCapture stopped. Total unique digits captured: {len(digit_arrays)}")
                break

            if(win32api.GetAsyncKeyState(0x23)&0x8001 > 0):
                if not key_pressed:
                    if not pause:
                        pause = True
                        print("\nPause!")
                    else:
                        pause = False
                        print("\nResume!")
                    key_pressed = True
            else:
                key_pressed = False

            if pause:
                time.sleep(0.1)
                continue

            image = camera.get_latest_frame()
            speedometer = image[757:793, 1015:1093]
            speedometer = cv2.cvtColor(speedometer, cv2.COLOR_BGR2GRAY)
            _, speedometer = cv2.threshold(speedometer, 254, 255, cv2.THRESH_BINARY)
            
            digits = {
                'hundreds': speedometer[0:35, 0:25],
                'tens': speedometer[0:35, 26:51],
                'ones': speedometer[0:35, 52:77]
            }
            
            saved_any = False
            for digit_type, digit_img in digits.items():
                if save_if_unique(digit_img, digit_type, counter):
                    saved_any = True
                    print(f"Saved new unique {digit_type} digit (total: {len(digit_arrays)})")
            
            if saved_any:
                counter += 1
            
    except KeyboardInterrupt:
        print(f"\nCapture stopped. Total unique digits captured: {len(digit_arrays)}")
    finally:
        camera.stop()