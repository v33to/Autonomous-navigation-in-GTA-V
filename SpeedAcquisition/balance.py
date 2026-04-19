import os
import numpy as np
import cv2
from pathlib import Path
import shutil
from skimage.metrics import structural_similarity as ssim

"""
Script that balances digit classes by either selecting diverse samples using similarity analysis (when over-represented) 
or duplicating existing samples (when under-represented) to achieve uniform class distribution.
"""

path = os.getcwd()
path += '/SpeedAcquisition/Digits'
input_path = path + '/labeled'
output_path = path + '/balanced'

def calculate_similarity_matrix(images):
    """
    Calculate similarity matrix between all images using SSIM.
    """
    n = len(images)
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            score = ssim(images[i], images[j], full=False)
            similarity_matrix[i, j] = score
            similarity_matrix[j, i] = score
        similarity_matrix[i, i] = 1.0
        
    return similarity_matrix

def select_diverse_samples(images, paths, n_samples):
    """
    Select diverse samples using similarity-based selection.
    """
    if len(images) <= n_samples:
        return paths
    
    similarity_matrix = calculate_similarity_matrix(images)
    selected_indices = [0]
    
    while len(selected_indices) < n_samples:
        avg_similarity = np.mean(similarity_matrix[selected_indices, :], axis=0)
        candidate_indices = list(set(range(len(images))) - set(selected_indices))
        next_idx = candidate_indices[np.argmin(avg_similarity[candidate_indices])]
        selected_indices.append(next_idx)
    
    return [paths[i] for i in selected_indices]

def balance_dataset(target_samples=100):
    """
    Balance the dataset by selecting diverse samples or duplicating existing ones.
    """
    os.makedirs(output_path, exist_ok=True)
    
    for class_name in os.listdir(input_path):
        class_dir = os.path.join(input_path, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        print(f"\nProcessing class: {class_name}")
        output_class_dir = os.path.join(output_path, class_name)
        os.makedirs(output_class_dir, exist_ok=True)
        
        images = []
        paths = []
        for img_path in Path(class_dir).glob('*.png'):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                paths.append(img_path)
        
        if not images:
            print(f"No images found in {class_dir}")
            continue
            
        print(f"Found {len(images)} original images")
        
        if len(images) > target_samples:
            print("Selecting diverse samples...")
            selected_paths = select_diverse_samples(images, paths, target_samples)
        else:
            print("Duplicating samples...")
            repetitions = int(np.ceil(target_samples / len(paths)))
            selected_paths = paths * repetitions
            selected_paths = selected_paths[:target_samples]
        
        for idx, src_path in enumerate(selected_paths):
            dst_path = os.path.join(output_class_dir, f"{class_name}_{idx:03d}.png")
            shutil.copy2(src_path, dst_path)
        
        print(f"Saved {len(selected_paths)} images")

def analyze_dataset(directory):
    """
    Print statistics about the dataset.
    """
    print(f"\nDataset Statistics for {directory}:")
    print("-" * 50)
    
    for class_name in sorted(os.listdir(directory)):
        class_dir = os.path.join(directory, class_name)
        if os.path.isdir(class_dir):
            n_images = len(list(Path(class_dir).glob('*.png')))
            print(f"{class_name}: {n_images} images")

if __name__ == "__main__":
    print("Original dataset statistics:")
    analyze_dataset(input_path)
    print("\nStarting dataset balancing...")
    balance_dataset(target_samples=100)
    print("\nBalanced dataset statistics:")
    analyze_dataset(output_path)