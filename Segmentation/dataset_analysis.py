import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import numpy as np
import os

'''
Script that analyzes image/label pairs, computes dataset statistics, extracts color mappings from label palettes 
and creates stratified train/validation splits while preserving class balance across pixel distributions.
'''

def iterative_train_test_split(indices, y, test_size=0.2):
    """
    Custom iterative stratification function for pixel distribution
    """
    n_samples = y.shape[0]
    n_test = int(n_samples * test_size)
    n_train = n_samples - n_test
    train_indices = []
    test_indices = []
    y_sum = y.sum(axis=0).astype(float)
    desired_test = y_sum * test_size
    desired_train = y_sum - desired_test
    count_test = np.zeros(y.shape[1], dtype=float)
    count_train = np.zeros(y.shape[1], dtype=float)
    
    for idx in np.random.permutation(indices):
        sample = y[idx].astype(float)
        test_deficit = np.maximum(desired_test - count_test, 0)
        train_deficit = np.maximum(desired_train - count_train, 0)
        test_contribution = np.sum(sample * test_deficit)
        train_contribution = np.sum(sample * train_deficit)
        if (test_contribution > train_contribution and len(test_indices) < n_test) or len(train_indices) >= n_train:
            test_indices.append(idx)
            count_test += sample
        else:
            train_indices.append(idx)
            count_train += sample

    return np.array(train_indices), np.array(test_indices)

def analyze_and_prepare_dataset(base_path, test_size=0.2, seed=42):
    """
    Function to analyze dataset and create stratified splits
    """
    image_dir = Path(base_path) / "processed_images"
    label_dir = Path(base_path) / "processed_labels"

    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(f"Paths not found: {image_dir}, {label_dir}")

    image_paths = sorted(image_dir.glob('*.png'))
    label_paths = sorted(label_dir.glob('*.png'))

    if len(image_paths) != len(label_paths):
        raise ValueError("Mismatch between number of images and labels")

    total_sum = torch.zeros(3, dtype=torch.float64)
    total_square_sum = torch.zeros(3, dtype=torch.float64)
    total_pixels = 0
    image_sizes = set()
    unique_labels = set()
    color_mapping = {}
    pixel_counts = []
    image_ids = []
    transform = transforms.ToTensor()
    first_label = Image.open(label_paths[0])
    base_palette = first_label.getpalette() if first_label.mode == 'P' else None

    print("Processing dataset...")
    for img_path, label_path in tqdm(zip(image_paths, label_paths), total=len(image_paths)):
        try:
            img = Image.open(img_path)
            img_tensor = transform(img)
            image_sizes.add(img.size)
            total_sum += img_tensor.sum(dim=[1, 2])
            total_square_sum += (img_tensor ** 2).sum(dim=[1, 2])
            total_pixels += img_tensor[0].numel()
            label_img = Image.open(label_path)
            label = np.array(label_img)
            image_ids.append(img_path.stem) 
            current_labels = np.unique(label)
            unique_labels.update(current_labels)
            
            if base_palette and label_img.mode == 'P':
                current_palette = label_img.getpalette()
                if current_palette != base_palette:
                    print(f"Warning: Palette mismatch in {label_path.name}")
            
            class_counts = {}
            for cls in current_labels:
                class_counts[cls] = np.sum(label == cls)
            pixel_counts.append(class_counts)
            
        except Exception as e:
            print(f"Error processing {img_path.name}: {str(e)}")
            continue

    mean = total_sum / total_pixels
    std = torch.sqrt((total_square_sum / total_pixels) - (mean ** 2))
    sorted_classes = sorted(unique_labels)
    n_classes = len(sorted_classes)
    pixel_matrix = np.zeros((len(image_ids), n_classes), dtype=np.int64)
    for i, counts in enumerate(pixel_counts):
        for cls, cnt in counts.items():
            cls_idx = sorted_classes.index(cls)
            pixel_matrix[i, cls_idx] = cnt

    epsilon = 1e-6
    class_freq = pixel_matrix.sum(axis=0)
    total_pixels = class_freq.sum()
    weights = total_pixels / (class_freq.astype(float) + epsilon)
    weights /= weights.mean()

    if base_palette:
        for cls in sorted_classes:
            if cls * 3 + 2 < len(base_palette):
                color_mapping[cls] = {
                    'rgb': (
                        base_palette[cls*3],
                        base_palette[cls*3+1],
                        base_palette[cls*3+2]
                    ),
                    'hex': f"#{base_palette[cls*3]:02x}{base_palette[cls*3+1]:02x}{base_palette[cls*3+2]:02x}"
                }
            else:
                color_mapping[cls] = {'rgb': (0, 0, 0), 'hex': '#000000'}
                print(f"Warning: Class {cls} exceeds palette length")

    np.random.seed(seed)
    indices = np.arange(len(image_ids))
    train_idx, val_idx = iterative_train_test_split(indices, pixel_matrix[:, 1:], test_size)

    with open(Path(base_path) / 'train_ids.txt', 'w') as f:
        for idx in train_idx:
            f.write(f"{image_ids[idx]}\n")
            
    with open(Path(base_path) / 'val_ids.txt', 'w') as f:
        for idx in val_idx:
            f.write(f"{image_ids[idx]}\n")

    report = [
        "=== Dataset Analysis Report ===",
        f"\nBasic Statistics:",
        f"Number of images: {len(image_paths)}",
        f"Image sizes found: {', '.join([f'{w}x{h}' for (w, h) in image_sizes])}",
        f"Mean (RGB): [{mean[0]:.3f}, {mean[1]:.3f}, {mean[2]:.3f}]",
        f"Std (RGB): [{std[0]:.3f}, {std[1]:.3f}, {std[2]:.3f}]",
        
        f"\nClass Information:",
        f"Number of unique classes: {len(sorted_classes)}",
        f"Class IDs: {', '.join(map(str, sorted_classes))}",
        
        "\nColor Mapping:",
    ]
    
    for cls in sorted_classes:
        if cls in color_mapping:
            colors = color_mapping[cls]
            report.append(
                f"Class {cls}: RGB{colors['rgb']} | HEX {colors['hex']}"
            )
    
    report.append("\nClass Distribution (percentage of total pixels):")
    for cls in sorted_classes:
        percentage = (class_freq[sorted_classes.index(cls)] / total_pixels) * 100
        report.append(
            f"Class {cls}: {percentage:.2f}% "
            f"({class_freq[sorted_classes.index(cls)]:,} pixels)"
        )
    
    report.append("\nClass Weights (inverse frequency):")
    for cls, weight in zip(sorted_classes, weights):
        report.append(f"Class {cls}: {weight:.4f}")
    report.append("\nDataset Splits:")
    report.append(f"Training samples: {len(train_idx)}")
    report.append(f"Validation samples: {len(val_idx)}")
    report.append("\nSplit files created: train_ids.txt, val_ids.txt")

    output_path = Path(base_path) / "dataset_report.txt"
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))
    print('\n'.join(report))
    print(f"\nReport saved to: {output_path}")

    return {
        'num_images': len(image_paths),
        'image_sizes': list(image_sizes),
        'mean': mean,
        'std': std,
        'unique_classes': sorted_classes,
        'class_distribution': class_freq,
        'color_mapping': color_mapping,
        'class_weights': weights
    }

if __name__ == "__main__":
    path = os.getcwd()
    path += '/Segmentation/'
    try:
        stats = analyze_and_prepare_dataset(path)
        print("\nProcessing complete. Report generated and splits created.")
    except Exception as e:
        print(f"Error: {str(e)}")