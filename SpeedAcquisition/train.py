import cv2
import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path

"""
SVM model training and validation that performs k-fold cross-validation, evaluates classification performance
and saves the final trained model for speedometer digit recognition.
"""

path = os.getcwd()
path += '/SpeedAcquisition/'

def load_balanced_dataset():
    """
    Load images from the balanced dataset directory
    """
    balanced_path = path + 'Digits/balanced'
    X = []
    y = []
    
    for class_name in os.listdir(balanced_path):
        class_dir = os.path.join(balanced_path, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        if class_name == 'black':
            label = 10
        elif class_name == 'progress_bar':
            label = 11
        else:
            label = int(class_name)
            
        for img_path in Path(class_dir).glob('*.png'):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                bit_array = img.ravel() >= 250
                processed_img = bit_array.astype(int)
                X.append(processed_img)
                y.append(label)
    
    return np.array(X), np.array(y)

def init():
    """
    Initialize SVM with parameters
    """
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
    return svm

def cross_validate(X, y, n_splits=5):
    """
    Perform cross-validation and return metrics
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_accuracies = []
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        print(f"\nFold {fold}/{n_splits}")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        svm = init()
        svm.train(X_train.astype(np.float32), cv2.ml.ROW_SAMPLE, y_train)
        _, y_pred = svm.predict(X_val.astype(np.float32))
        
        accuracy = accuracy_score(y_val, y_pred)
        fold_accuracies.append(accuracy)
        class_names = [str(i) for i in range(12)]
        report = classification_report(y_val, y_pred, target_names=class_names)
        fold_metrics.append(report)
        print(f"Fold {fold} Accuracy: {accuracy * 100:.2f}%")
        print("\nClassification Report:")
        print(report)
    
    return fold_accuracies, fold_metrics

if __name__ == "__main__":
    print("Loading balanced dataset...")
    X, y = load_balanced_dataset()
    
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    print(f"Loaded {len(X)} images")
    print("Class distribution:")
    for i in range(12):
        count = np.sum(y == i)
        class_name = 'black' if i == 10 else 'progress_bar' if i == 11 else str(i)
        print(f"Class {class_name}: {count} images")
    
    print("\nStarting cross-validation...")
    accuracies, metrics = cross_validate(X, y)
    print("\nCross-validation Summary:")
    print(f"Mean Accuracy: {np.mean(accuracies) * 100:.2f}%")
    print(f"Std Accuracy: {np.std(accuracies) * 100:.2f}%")
    
    with open(path + 'cross_validation_results.txt', 'w') as f:
        f.write("Cross-validation Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Mean Accuracy: {np.mean(accuracies) * 100:.2f}%\n")
        f.write(f"Std Accuracy: {np.std(accuracies) * 100:.2f}%\n\n")
        
        for fold, (acc, report) in enumerate(zip(accuracies, metrics), 1):
            f.write(f"\nFold {fold}\n")
            f.write("-" * 20 + "\n")
            f.write(f"Accuracy: {acc * 100:.2f}%\n")
            f.write("Classification Report:\n")
            f.write(report)
            f.write("\n")
    
    print("\nTraining final model on full dataset...")
    svm = init()
    svm.train(X.astype(np.float32), cv2.ml.ROW_SAMPLE, y)
    svm.save(path + 'svm.yml')
    print("Final model saved as 'svm.yml'")