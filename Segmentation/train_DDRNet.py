import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import datetime
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchinfo import summary
import matplotlib.pyplot as plt
import json

"""
DDRNet-23-slim training script with mixed precision, checkpoint management and evaluation metrics.
Includes TorchScript model saving for deployment.
"""

path = os.getcwd()
path += '/Segmentation/'

def get_class_info():
    return {
        0: {"color": (0, 0, 0), "name": "unlabeled + ego vehicle + static + fence"},
        1: {"color": (128, 64, 128), "name": "road + ground"},
        2: {"color": (244, 35, 232), "name": "sidewalk"},
        3: {"color": (70, 70, 70), "name": "building + dynamic + wall + bridge + tunnel + guard rail"},
        4: {"color": (153, 153, 153), "name": "pole"},
        5: {"color": (250, 170, 30), "name": "traffic light + traffic sign"},
        6: {"color": (107, 142, 35), "name": "vegetation"},
        7: {"color": (152, 251, 152), "name": "terrain"},
        8: {"color": (70, 130, 180), "name": "sky"},
        9: {"color": (220, 20, 60), "name": "person + rider"},
        10: {"color": (0, 0, 142), "name": "car + truck + bus + trailer + train + motorcycle + bicycle"}
    }

def apply_color_mapping(label_img):
    """
    Converts grayscale label masks to RGB visualization using predefined class colors.
    """
    class_info = get_class_info()
    color_map = {class_id: info["color"] for class_id, info in class_info.items()}
    height, width = label_img.shape
    colored_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    for label, color in color_map.items():
        mask = label_img == label
        colored_img[mask] = color
    return colored_img

def save_model_info(model_path, dataset, model, final_metrics):
    """
    Save model performance and config to a text file
    """
    class_info = get_class_info()
    
    with open(model_path, 'w') as f:
        f.write(f"Model saved on: {datetime.datetime.now()}\n\n")
        f.write("Dataset Statistics:\n")
        f.write(f"Mean: {dataset.mean}\n")
        f.write(f"Std: {dataset.std}\n\n")
        f.write("Final Metrics Per Class:\n")
        f.write("Class ID\tIoU\tDice\tClass Name\n")
        f.write("-" * 70 + "\n")
        
        for cls in range(1, model.n_classes):
            iou = final_metrics['class_ious'][cls].item()
            dice = final_metrics['class_dice'][cls].item()
            class_name = class_info[cls]["name"]
            f.write(f"{cls}\t{iou:.4f}\t{dice:.4f}\t{class_name}\n")
        
        f.write("\nOverall Metrics:\n")
        f.write(f"Mean IoU: {final_metrics['class_ious'].mean():.4f}\n")
        f.write(f"Mean Dice: {final_metrics['class_dice'].mean():.4f}\n")

def save_predictions_as_imgs(loader, model, folder=path + "saved_images/", device="cuda", num_images=10):
    """
    Saves side-by-side visualizations of original images, ground truth and model predictions
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    model.eval()
    count = 0
    
    with torch.no_grad():
        for _, (x, y) in enumerate(loader):
            if count >= num_images:
                break
                
            x = x.to(device=device)
            with autocast(device_type='cuda', dtype=torch.float16):
                preds = model(x)
                # Handle deep supervision output (list)
                if isinstance(preds, list):
                    preds = preds[0]
            
            preds = torch.argmax(preds, dim=1)
            preds = preds.cpu().numpy()
            orig_images = x.cpu().numpy()
            y = y.cpu().numpy()
            
            for i in range(min(preds.shape[0], num_images - count)):
                plt.figure(figsize=(15, 5))
                plt.subplot(1, 3, 1)
                plt.title('Original Image')
                mean = np.array([0.383, 0.376, 0.358])[:, None, None]
                std = np.array([0.221, 0.209, 0.190])[:, None, None]
                orig_img = orig_images[i] * std + mean
                orig_img = orig_img.transpose(1, 2, 0)
                orig_img = np.clip(orig_img, 0, 1)
                plt.imshow(orig_img)
                plt.axis('off')
                plt.subplot(1, 3, 2)
                plt.title('Ground Truth')
                gt_img_colored = apply_color_mapping(y[i])
                plt.imshow(gt_img_colored)
                plt.axis('off')
                plt.subplot(1, 3, 3)
                plt.title('Prediction')
                pred_img_colored = apply_color_mapping(preds[i])
                plt.imshow(pred_img_colored)
                plt.axis('off')
                plt.savefig(f"{folder}/combined_{count}.png", bbox_inches='tight', dpi=300)
                plt.close()
                count += 1
                if count >= num_images:
                    break
    
    model.train()

class SegmentationDataset(Dataset):
    """
    Custom dataset for loading and preprocessing segmentation image/label pairs with class mapping
    """
    def __init__(self, image_dir, label_dir):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.mean = [0.383, 0.376, 0.358]
        self.std = [0.221, 0.209, 0.190]
        self.original_classes = [0, 7, 8, 11, 17, 19, 21, 22, 23, 24, 26]
        self.class_mapping = {original: idx for idx, original in enumerate(self.original_classes)}
        self.images = sorted(list(self.image_dir.glob('*.png')))
        self.labels = sorted(list(self.label_dir.glob('*.png')))
        
        self.transform = A.Compose([
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2(),
        ], additional_targets={'mask': 'mask'})
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = np.array(Image.open(self.images[idx]))
        label = np.array(Image.open(self.labels[idx]))
        transformed = self.transform(image=img, mask=label)
        img = transformed['image']
        label = transformed['mask']
        mapped_label = np.zeros_like(label)

        for original, mapped in self.class_mapping.items():
            mapped_label[label == original] = mapped
        return img, torch.from_numpy(mapped_label).long()

def calculate_metrics(pred, target, n_classes, device):
    """
    Computes IoU, Dice coefficient and pixel accuracy for each class and overall performance evaluation.
    """
    pred = pred.argmax(dim=1)
    metrics = {}
    class_ious = torch.zeros(n_classes, device=device)
    class_dice = torch.zeros(n_classes, device=device)
    
    for cls in range(1, n_classes):
        pred_mask = (pred == cls)
        target_mask = (target == cls)
        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()
        iou = intersection / (union + 1e-6)
        class_ious[cls] = iou
        dice = (2. * intersection) / (pred_mask.sum() + target_mask.sum() + 1e-6)
        class_dice[cls] = dice
    
    metrics['mean_iou'] = class_ious.mean()
    metrics['class_ious'] = class_ious
    metrics['mean_dice'] = class_dice.mean()
    metrics['class_dice'] = class_dice
    correct = (pred == target).sum().float()
    total = torch.numel(target)
    metrics['pixel_accuracy'] = correct / total
    return metrics

# DDRNet-23-slim architecture components
bn_mom = 0.1
BatchNorm2d = nn.BatchNorm2d

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        if self.no_relu:
            return out
        else:
            return self.relu(out)

class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.no_relu:
            return out
        else:
            return self.relu(out)

class DAPPM(nn.Module):
    """Deep Aggregation Pyramid Pooling Module"""
    def __init__(self, inplanes, branch_planes, outplanes):
        super(DAPPM, self).__init__()
        self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale3 = nn.Sequential(nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale0 = nn.Sequential(
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.process1 = nn.Sequential(
                                    BatchNorm2d(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process2 = nn.Sequential(
                                    BatchNorm2d(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process3 = nn.Sequential(
                                    BatchNorm2d(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process4 = nn.Sequential(
                                    BatchNorm2d(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )        
        self.compression = nn.Sequential(
                                    BatchNorm2d(branch_planes * 5, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
                                    )
        self.shortcut = nn.Sequential(
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
                                    )

    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]        
        x_list = []

        x_list.append(self.scale0(x))
        x_list.append(self.process1((nn.functional.interpolate(self.scale1(x),
                        size=[height, width],
                        mode='bilinear', align_corners=False)+x_list[0])))
        x_list.append((self.process2((nn.functional.interpolate(self.scale2(x),
                        size=[height, width],
                        mode='bilinear', align_corners=False)+x_list[1]))))
        x_list.append(self.process3((nn.functional.interpolate(self.scale3(x),
                        size=[height, width],
                        mode='bilinear', align_corners=False)+x_list[2])))
        x_list.append(self.process4((nn.functional.interpolate(self.scale4(x),
                        size=[height, width],
                        mode='bilinear', align_corners=False)+x_list[3])))
       
        out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
        return out 

class SegmentHead(nn.Module):
    """Segmentation head for final predictions"""
    def __init__(self, inplanes, interplanes, outplanes, scale_factor=8):
        super(SegmentHead, self).__init__()
        self.bn1 = BatchNorm2d(inplanes, momentum=bn_mom)
        self.conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm2d(interplanes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))

        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = nn.functional.interpolate(out,
                        size=[height, width],
                        mode='bilinear', align_corners=False)

        return out

class DDRNet23Slim(nn.Module):
    """
    DDRNet-23-slim for semantic segmentation
    """
    def __init__(self, num_classes=19, planes=32, spp_planes=128, head_planes=64, augment=False):
        super(DDRNet23Slim, self).__init__()

        highres_planes = planes * 2
        self.augment = augment
        self.n_classes = num_classes

        # Initial downsampling
        self.conv1 = nn.Sequential(
                          nn.Conv2d(3, planes, kernel_size=3, stride=2, padding=1),
                          BatchNorm2d(planes, momentum=bn_mom),
                          nn.ReLU(inplace=True),
                          nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1),
                          BatchNorm2d(planes, momentum=bn_mom),
                          nn.ReLU(inplace=True),
                      )

        self.relu = nn.ReLU(inplace=False)
        
        # Encoder layers
        self.layer1 = self._make_layer(BasicBlock, planes, planes, 2)
        self.layer2 = self._make_layer(BasicBlock, planes, planes * 2, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, planes * 2, planes * 4, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, planes * 4, planes * 8, 2, stride=2)

        # Compression layers for bilateral fusion
        self.compression3 = nn.Sequential(
                                          nn.Conv2d(planes * 4, highres_planes, kernel_size=1, bias=False),
                                          BatchNorm2d(highres_planes, momentum=bn_mom),
                                          )

        self.compression4 = nn.Sequential(
                                          nn.Conv2d(planes * 8, highres_planes, kernel_size=1, bias=False),
                                          BatchNorm2d(highres_planes, momentum=bn_mom),
                                          )

        # Downsampling for high-to-low fusion
        self.down3 = nn.Sequential(
                                   nn.Conv2d(highres_planes, planes * 4, kernel_size=3, stride=2, padding=1, bias=False),
                                   BatchNorm2d(planes * 4, momentum=bn_mom),
                                   )

        self.down4 = nn.Sequential(
                                   nn.Conv2d(highres_planes, planes * 4, kernel_size=3, stride=2, padding=1, bias=False),
                                   BatchNorm2d(planes * 4, momentum=bn_mom),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(planes * 4, planes * 8, kernel_size=3, stride=2, padding=1, bias=False),
                                   BatchNorm2d(planes * 8, momentum=bn_mom),
                                   )

        # High-resolution branch layers
        self.layer3_ = self._make_layer(BasicBlock, planes * 2, highres_planes, 2)
        self.layer4_ = self._make_layer(BasicBlock, highres_planes, highres_planes, 2)
        self.layer5_ = self._make_layer(Bottleneck, highres_planes, highres_planes, 1)

        # Low-resolution branch bottleneck
        self.layer5 = self._make_layer(Bottleneck, planes * 8, planes * 8, 1, stride=2)

        # DAPPM module
        self.spp = DAPPM(planes * 16, spp_planes, planes * 4)

        # Auxiliary head for deep supervision (training only)
        if self.augment:
            self.seghead_extra = SegmentHead(highres_planes, head_planes, num_classes, scale_factor=8)

        # Final segmentation head
        self.final_layer = SegmentHead(planes * 4, head_planes, num_classes, scale_factor=8)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == (blocks-1):
                layers.append(block(inplanes, planes, stride=1, no_relu=True))
            else:
                layers.append(block(inplanes, planes, stride=1, no_relu=False))

        return nn.Sequential(*layers)

    def forward(self, x):
        width_output = x.shape[-1] // 8
        height_output = x.shape[-2] // 8
        layers = []

        x = self.conv1(x)

        x = self.layer1(x)
        layers.append(x)

        x = self.layer2(self.relu(x))
        layers.append(x)
  
        x = self.layer3(self.relu(x))
        layers.append(x)
        x_ = self.layer3_(self.relu(layers[1]))

        x = x + self.down3(self.relu(x_))
        x_ = x_ + nn.functional.interpolate(
                        self.compression3(self.relu(layers[2])),
                        size=[height_output, width_output],
                        mode='bilinear', align_corners=False)
        
        # Store for auxiliary head
        temp = x_

        x = self.layer4(self.relu(x))
        layers.append(x)
        x_ = self.layer4_(self.relu(x_))

        x = x + self.down4(self.relu(x_))
        x_ = x_ + nn.functional.interpolate(
                        self.compression4(self.relu(layers[3])),
                        size=[height_output, width_output],
                        mode='bilinear', align_corners=False)

        x_ = self.layer5_(self.relu(x_))
        x = nn.functional.interpolate(
                        self.spp(self.layer5(self.relu(x))),
                        size=[height_output, width_output],
                        mode='bilinear', align_corners=False)

        x_ = self.final_layer(x + x_)

        if self.augment: 
            x_extra = self.seghead_extra(temp)
            return [x_, x_extra]
        else:
            return x_

def save_training_stats(stats_file, epoch_stats):
    """
    Save training statistics to a JSON file
    """
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            stats = json.load(f)
    else:
        stats = {
            'epochs': [],
            'train_loss': [],
            'val_iou': [],
            'val_dice': [],
            'val_accuracy': [],
            'learning_rate': [],
            'class_metrics': []
        }
    
    stats['epochs'].append(epoch_stats['epoch'])
    stats['train_loss'].append(epoch_stats['train_loss'])
    stats['val_iou'].append(epoch_stats['val_iou'])
    stats['val_dice'].append(epoch_stats['val_dice'])
    stats['val_accuracy'].append(epoch_stats['val_accuracy'])
    stats['learning_rate'].append(epoch_stats['learning_rate'])
    
    class_metrics = {
        'epoch': epoch_stats['epoch'],
        'class_ious': [float(iou) for iou in epoch_stats['class_ious']],
        'class_dice': [float(dice) for dice in epoch_stats['class_dice']]
    }
    stats['class_metrics'].append(class_metrics)
    
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=4)

def train_model(train_image_dir, train_label_dir, val_image_dir, val_label_dir, num_epochs=200, batch_size=32, patience=10):
    """
    Main training loop with mixed precision, checkpointing, early stopping and evaluation
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    best_model_path = path + 'segmentation_ddrnet.pt'
    best_model_dict_path = path + 'segmentation_ddrnet_dict.pt'
    checkpoint_path = path + 'segmentation_ddrnet_checkpoint.pt'
    stats_file = path + 'training_stats_ddrnet.json'
    
    os.makedirs(os.path.dirname(stats_file), exist_ok=True)
    
    train_dataset = SegmentationDataset(train_image_dir, train_label_dir)
    val_dataset = SegmentationDataset(val_image_dir, val_label_dir)
    n_classes = len(train_dataset.class_mapping)

    # Initialize DDRNet-23-slim with deep supervision for training
    model = DDRNet23Slim(num_classes=n_classes, planes=32, spp_planes=128, head_planes=64, augment=True).to(device)
    model.n_classes = n_classes
    
    # Load Cityscapes pre-trained weights
    cityscapes_path = path + 'best_val_smaller.pth'
    if os.path.exists(cityscapes_path):
        print(f"Loading Cityscapes pre-trained weights from {cityscapes_path}")
        try:
            checkpoint = torch.load(cityscapes_path, map_location='cpu', weights_only=False)
            
            # The checkpoint is a flat dictionary with 'model.' and 'loss.' prefixes
            # Filter out non-model keys (like 'loss.criterion.weight')
            pretrained_dict = {k: v for k, v in checkpoint.items() if k.startswith('model.')}
            
            # Remove 'model.' prefix to match our model's state dict
            pretrained_dict = {k.replace('model.', ''): v for k, v in pretrained_dict.items()}
            
            # Get current model state
            model_dict = model.state_dict()
            
            # Filter out layers that won't be compatible:
            # 1. Segmentation heads (final_layer and seghead_extra) have different output classes (19 vs 11)
            # 2. Only load backbone and feature extraction layers
            excluded_keys = ['final_layer', 'seghead_extra']
            
            compatible_dict = {}
            skipped_layers = []
            
            for k, v in pretrained_dict.items():
                # Skip if key contains excluded prefixes
                if any(excluded in k for excluded in excluded_keys):
                    skipped_layers.append(f"{k} (segmentation head - different num classes)")
                    continue
                
                # Check if layer exists in current model and has matching shape
                if k in model_dict:
                    if model_dict[k].shape == v.shape:
                        compatible_dict[k] = v
                    else:
                        skipped_layers.append(f"{k} (shape mismatch: {v.shape} vs {model_dict[k].shape})")
                else:
                    skipped_layers.append(f"{k} (not in current model)")
            
            # Update model with compatible weights
            model_dict.update(compatible_dict)
            model.load_state_dict(model_dict, strict=False)
            
            print(f"✓ Successfully loaded {len(compatible_dict)} pre-trained layers from Cityscapes")
            print(f"✗ Skipped {len(skipped_layers)} incompatible layers")
            
            if len(skipped_layers) <= 15:  # Print first few skipped layers
                print("\nSkipped layers (segmentation heads will be trained from scratch):")
                for layer in skipped_layers[:15]:
                    print(f"  - {layer}")
                if len(skipped_layers) > 15:
                    print(f"  ... and {len(skipped_layers) - 15} more")
                    
        except Exception as e:
            print(f"Warning: Could not load pre-trained weights: {e}")
            print("Training from scratch...")
    else:
        print(f"Cityscapes pre-trained weights not found at {cityscapes_path}. Training from scratch...")
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    class_weights = torch.ones(model.n_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=0)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.1,
        patience=5,
        min_lr=1e-6
    )
    
    scaler = GradScaler()
    
    start_epoch = 0
    best_metric = 0.0
    patience_counter = 0
    
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_metric = checkpoint['best_metric']
            patience_counter = checkpoint['patience_counter']
            print(f"Successfully loaded checkpoint from epoch {checkpoint['epoch']}")
            print(f"Previous best metric: {best_metric:.4f}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting with fresh model.")
    else:
        print("No previous checkpoint found. Starting with fresh model.")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print("\nModel Architecture Summary:")
    summary(model, input_size=(batch_size, 3, 256, 512))
    
    try:
        for epoch in range(start_epoch, num_epochs):
            model.train()
            epoch_loss = torch.tensor(0.0, device=device)
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
            
            for _, (images, labels) in enumerate(progress_bar):
                images = images.to(device)
                labels = labels.to(device)
                
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(images)
                    # Handle deep supervision (auxiliary loss)
                    if isinstance(outputs, list):
                        # Main loss + auxiliary loss with weight 0.4
                        loss = criterion(outputs[0], labels) + 0.4 * criterion(outputs[1], labels)
                        outputs = outputs[0]
                    else:
                        loss = criterion(outputs, labels)
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss
                progress_bar.set_postfix({'loss': f'{loss:.4f}'})
            
            avg_train_loss = epoch_loss / len(train_loader)
            
            model.eval()
            val_iou = torch.tensor(0.0, device=device)
            val_dice = torch.tensor(0.0, device=device)
            val_accuracy = torch.tensor(0.0, device=device)
            val_class_ious = torch.zeros(model.n_classes, device=device)
            val_class_dice = torch.zeros(model.n_classes, device=device)
            n_batches = len(val_loader)
            
            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc='Validation'):
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    with autocast(device_type='cuda', dtype=torch.float16):
                        outputs = model(images)
                        # Use only main output for validation
                        if isinstance(outputs, list):
                            outputs = outputs[0]
                    
                    metrics = calculate_metrics(outputs, labels, model.n_classes, device)
                    val_iou += metrics['mean_iou']
                    val_dice += metrics['mean_dice']
                    val_accuracy += metrics['pixel_accuracy']
                    val_class_ious += metrics['class_ious']
                    val_class_dice += metrics['class_dice']
            
            mean_metrics = {
                'iou': val_iou / n_batches,
                'dice': val_dice / n_batches,
                'accuracy': val_accuracy / n_batches,
                'class_ious': val_class_ious / n_batches,
                'class_dice': val_class_dice / n_batches
            }
            
            current_lr = optimizer.param_groups[0]['lr']
            
            epoch_stats = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss.item(),
                'val_iou': mean_metrics['iou'].item(),
                'val_dice': mean_metrics['dice'].item(),
                'val_accuracy': mean_metrics['accuracy'].item(),
                'learning_rate': current_lr,
                'class_ious': mean_metrics['class_ious'].cpu().tolist(),
                'class_dice': mean_metrics['class_dice'].cpu().tolist()
            }
            save_training_stats(stats_file, epoch_stats)
            
            final_metrics = {
                'class_ious': mean_metrics['class_ious'].cpu(),
                'class_dice': mean_metrics['class_dice'].cpu()
            }
            
            current_metric = (mean_metrics['iou'] + mean_metrics['dice']) / 2
            scheduler.step(current_metric)
            
            print(f"\nEpoch {epoch+1} Metrics:")
            print(f"Loss: {avg_train_loss:.4f}")
            print(f"Mean IoU: {mean_metrics['iou']:.4f}")
            print(f"Mean Dice: {mean_metrics['dice']:.4f}")
            print(f"Pixel Accuracy: {mean_metrics['accuracy']:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")
            
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'epoch': epoch,
                'best_metric': best_metric,
                'patience_counter': patience_counter,
                'n_classes': n_classes
            }
            torch.save(checkpoint, checkpoint_path)
            
            if current_metric > best_metric:
                best_metric = current_metric
                model.eval()
                
                # Save TorchScript model using trace
                try:
                    # Temporarily disable augment mode for consistent return type
                    original_augment = model.augment
                    model.augment = False
                    
                    # Create example input for tracing
                    example_input = torch.randn(1, 3, 256, 512).to(device)
                    
                    # Use trace instead of script to avoid return type issues
                    with torch.no_grad():
                        traced_model = torch.jit.trace(model, example_input)
                    
                    torch.jit.save(traced_model, best_model_path)
                    print(f"Saved best TorchScript model with metric: {best_metric:.4f}")
                    
                    # Restore augment mode
                    model.augment = original_augment
                except Exception as e:
                    print(f"Error saving TorchScript model: {e}")
                    # Make sure to restore augment mode even if error occurs
                    if 'original_augment' in locals():
                        model.augment = original_augment
                
                # Save state dictionary
                try:
                    torch.save(model.state_dict(), best_model_dict_path)
                    print(f"Saved best model state dictionary with metric: {best_metric:.4f}")
                except Exception as e:
                    print(f"Error saving state dictionary: {e}")
                
                save_predictions_as_imgs(
                    val_loader,
                    model,
                    folder=f"{path}saved_images/",
                    device=device
                )
                save_model_info(path + 'info_ddrnet.txt', train_dataset, model, final_metrics)
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered. No improvement for {patience} epochs.")
                break
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    TRAIN_IMAGE_DIR = path + "train_images"
    TRAIN_LABEL_DIR = path + "train_labels"
    VAL_IMAGE_DIR = path + "val_images"
    VAL_LABEL_DIR = path + "val_labels"
    train_model(TRAIN_IMAGE_DIR, TRAIN_LABEL_DIR, VAL_IMAGE_DIR, VAL_LABEL_DIR)