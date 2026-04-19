import os
import shutil

def organize_dataset(base_path, source_images, source_labels, train_ids_file, val_ids_file):
    """
    Organize dataset into train and validation sets.
    """
    with open(train_ids_file, 'r') as f:
        train_ids = [int(line.strip()) for line in f]
    with open(val_ids_file, 'r') as f:
        val_ids = [int(line.strip()) for line in f]

    directories = ['train_images', 'train_labels', 'val_images', 'val_labels']
    for directory in directories:
        dir_path = os.path.join(base_path, directory)
        os.makedirs(dir_path, exist_ok=True)
    
    stats = {
        'train_images_copied': 0,
        'train_labels_copied': 0,
        'val_images_copied': 0,
        'val_labels_copied': 0
    }
    
    for is_train in [True, False]:
        ids = train_ids if is_train else val_ids
        prefix = 'train' if is_train else 'val'
        
        for id_num in ids:
            padded_filename = f"{id_num:05d}.png"
            src_img = os.path.join(source_images, padded_filename)
            dst_img = os.path.join(base_path, f"{prefix}_images", padded_filename)
            src_label = os.path.join(source_labels, padded_filename)
            dst_label = os.path.join(base_path, f"{prefix}_labels", padded_filename)

            if os.path.exists(src_img):
                shutil.copy(src_img, dst_img)
                stats[f'{prefix}_images_copied'] += 1
            
            if os.path.exists(src_label):
                shutil.copy(src_label, dst_label)
                stats[f'{prefix}_labels_copied'] += 1
    
    print("\nDataset Split Statistics:")
    print(f"Total IDs in train set: {len(train_ids)}")
    print(f"Total IDs in validation set: {len(val_ids)}")
    print("\nActual files copied:")
    print(f"Training images copied: {stats['train_images_copied']}")
    print(f"Training labels copied: {stats['train_labels_copied']}")
    print(f"Validation images copied: {stats['val_images_copied']}")
    print(f"Validation labels copied: {stats['val_labels_copied']}")
    
    if stats['train_images_copied'] != stats['train_labels_copied']:
        print("\nWarning: Mismatch in training set!")
        print(f"Training images ({stats['train_images_copied']}) != Training labels ({stats['train_labels_copied']})")
    
    if stats['val_images_copied'] != stats['val_labels_copied']:
        print("\nWarning: Mismatch in validation set!")
        print(f"Validation images ({stats['val_images_copied']}) != Validation labels ({stats['val_labels_copied']})")

if __name__ == "__main__":
    path = os.getcwd()
    path += '/Segmentation/'
    source_images = os.path.join(path, "processed_images")
    source_labels = os.path.join(path, "processed_labels")
    train_ids_file = os.path.join(path, "train_ids.txt")
    val_ids_file = os.path.join(path, "val_ids.txt")
    organize_dataset(path, source_images, source_labels, train_ids_file, val_ids_file)