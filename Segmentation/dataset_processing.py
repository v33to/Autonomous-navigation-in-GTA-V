import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import random

'''
Script that resizes images to target dimensions, extracts ROI, applies final resizing and remaps semantic classes to consolidate similar categories.
It requires the GTAV dataset that can be downloaded from: https://download.visinf.tu-darmstadt.de/data/from_games/
'''

def process_and_save_dataset(images_dir, labels_dir, output_images_dir, output_labels_dir, target_size, roi, final_size):
    """
    Process and save dataset of images and labels with direct resizing, ROI extraction and class remapping
    """
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    print(f"Found {len(image_files)} images and {len(label_files)} labels")
    
    class_mapping = {
        0: 0,
        1: 0,
        4: 0,
        5: 11,
        6: 7,
        7: 7,
        8: 8,
        11: 11,
        12: 11,
        13: 0,
        14: 11,
        15: 11,
        16: 11,
        17: 17,
        19: 19,
        20: 19,
        21: 21,
        22: 22,
        23: 23,
        24: 24,
        25: 24,
        26: 26,
        27: 26,
        28: 26,
        30: 26,
        31: 26,
        32: 26,
        33: 26,
        34: 26,
    }

    max_class = 34
    lookup_table = np.arange(max_class + 1, dtype=np.uint8)
    for original, target in class_mapping.items():
        lookup_table[original] = target
    corrupt_files = []
    mismatched_dimensions = []
    missing_pairs = []
    processed_count = 0
    target_width, target_height = target_size
    roi_x1, roi_y1, roi_x2, roi_y2 = roi
    roi_width = roi_x2 - roi_x1
    roi_height = roi_y2 - roi_y1
    final_width, final_height = final_size
    
    print("\nProcessing Parameters:")
    print(f"Initial target size: {target_width}x{target_height}")
    print(f"ROI coordinates: ({roi_x1}, {roi_y1}) to ({roi_x2}, {roi_y2})")
    print(f"ROI size: {roi_width}x{roi_height}")
    print(f"Final output size: {final_width}x{final_height}")
    print("\nProcessing and saving dataset...")

    for img_file in tqdm(image_files):
        label_file = img_file
        if label_file not in label_files:
            missing_pairs.append(img_file)
            continue
        img_path = os.path.join(images_dir, img_file)
        label_path = os.path.join(labels_dir, label_file)

        try:
            image = Image.open(img_path)
            label = Image.open(label_path)
            image.load()
            label.load()
            
            if image.size != label.size:
                mismatched_dimensions.append({
                    'file': img_file,
                    'image_size': image.size,
                    'label_size': label.size,
                    'image_path': img_path,
                    'label_path': label_path
                })
            
            resized_image = image.resize(target_size, Image.Resampling.LANCZOS)
            resized_label = label.resize(target_size, Image.Resampling.NEAREST)
            roi_image = resized_image.crop((roi_x1, roi_y1, roi_x2, roi_y2))
            roi_label = resized_label.crop((roi_x1, roi_y1, roi_x2, roi_y2))
            final_image = roi_image.resize(final_size, Image.Resampling.LANCZOS)
            final_label = roi_label.resize(final_size, Image.Resampling.NEAREST)
            label_array = np.array(final_label)
            mapped_array = lookup_table[label_array]
            
            mapped_image = Image.fromarray(mapped_array, mode='P')
            original_palette = final_label.getpalette()
            new_palette = list(original_palette)
            
            for original, target in class_mapping.items():
                target_r = new_palette[target * 3]
                target_g = new_palette[target * 3 + 1]
                target_b = new_palette[target * 3 + 2]
                new_palette[original * 3] = target_r
                new_palette[original * 3 + 1] = target_g
                new_palette[original * 3 + 2] = target_b

            mapped_image.putpalette(new_palette)
            output_image_path = os.path.join(output_images_dir, img_file)
            output_label_path = os.path.join(output_labels_dir, label_file)
            final_image.save(output_image_path)
            mapped_image.save(output_label_path)
            processed_count += 1
            
        except (IOError, SyntaxError, ValueError) as e:
            corrupt_files.append((img_file, str(e)))
            continue
        except KeyboardInterrupt:
            print("\nProcess interrupted by user")
            break
    
    print("\nProcessing Summary:")
    print(f"Total images processed and saved: {processed_count}")
    print(f"Corrupt files found: {len(corrupt_files)}")
    print(f"Dimension mismatches found: {len(mismatched_dimensions)}")
    print(f"Missing pairs found: {len(missing_pairs)}")
    
    if corrupt_files:
        print("\nCorrupt files:")
        for file, error in corrupt_files:
            print(f"- {file}: {error}")
            
    if mismatched_dimensions:
        print("\nDimension mismatches (showing up to 10 random examples):")
        sample_size = min(10, len(mismatched_dimensions))
        random_samples = random.sample(mismatched_dimensions, sample_size)
        for item in random_samples:
            print(f"- {item['file']}")
            print(f"  Image size: {item['image_size']}")
            print(f"  Label size: {item['label_size']}")
            print(f"  Image path: {item['image_path']}")
            print(f"  Label path: {item['label_path']}")
            print()
            
    if missing_pairs:
        print("\nMissing pairs:")
        for file in missing_pairs:
            print(f"- {file}")
            
    print(f"\nProcessed images saved to: {output_images_dir}")
    print(f"Processed labels saved to: {output_labels_dir}")
    return processed_count, corrupt_files, mismatched_dimensions, missing_pairs

if __name__ == "__main__":
    path = os.getcwd()
    path += '/Segmentation/'
    images_dir = path + "images"
    labels_dir = path + "labels"
    output_images_dir = path + "processed_images"
    output_labels_dir = path + "processed_labels"
    target_size = (1152, 864)
    roi = (0, 117, 1152, 693)
    final_size = (512, 256)
    processed_count, corrupt_files, mismatched_dimensions, missing_pairs = process_and_save_dataset(
        images_dir, labels_dir,
        output_images_dir, output_labels_dir,
        target_size, roi, final_size
    )