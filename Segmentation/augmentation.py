import os
from PIL import Image
import glob

def augment_dataset(image_path, label_path):
    """
    Augment the dataset by creating horizontally flipped and rotated (-10 and +10 degrees) versions of all images and labels.
    """
    image_files = glob.glob(os.path.join(image_path, "*"))
    transformations = [
        {
            'suffix': '_flipped',
            'image_transform': lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),
            'label_transform': lambda lbl: lbl.transpose(Image.FLIP_LEFT_RIGHT)
        },
        {
            'suffix': '_rotated_-10',
            'image_transform': lambda img: img.rotate(-10, resample=Image.BILINEAR),
            'label_transform': lambda lbl: lbl.rotate(-10, resample=Image.NEAREST)
        },
        {
            'suffix': '_rotated_10',
            'image_transform': lambda img: img.rotate(10, resample=Image.BILINEAR),
            'label_transform': lambda lbl: lbl.rotate(10, resample=Image.NEAREST)
        }
    ]

    for img_file in image_files:
        filename = os.path.basename(img_file)
        label_file = os.path.join(label_path, filename)
        if not os.path.exists(label_file):
            print(f"Warning: No corresponding label found for {filename}")
            continue
            
        try:
            image = Image.open(img_file)
            label = Image.open(label_file)
            
            for transform in transformations:
                transformed_image = transform['image_transform'](image)
                transformed_label = transform['label_transform'](label)
                base_name, ext = os.path.splitext(filename)
                new_image_name = f"{base_name}{transform['suffix']}{ext}"
                new_label_name = f"{base_name}{transform['suffix']}{ext}"
                new_image_path = os.path.join(image_path, new_image_name)
                new_label_path = os.path.join(label_path, new_label_name)
                
                if os.path.exists(new_image_path) or os.path.exists(new_label_path):
                    print(f"Warning: Augmented files for {filename} with suffix {transform['suffix']} already exist. Skipping.")
                    continue
                
                transformed_image.save(new_image_path)
                transformed_label.save(new_label_path)
                print(f"Created {transform['suffix']} version of {filename}")
                
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    path = os.getcwd()
    path += '/Segmentation/'
    image_path = os.path.join(path, "train_images")
    label_path = os.path.join(path, "train_labels")
    print("Starting dataset augmentation...")
    augment_dataset(image_path, label_path)
    print("Augmentation completed!")