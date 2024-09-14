import os
import shutil
from sklearn.model_selection import train_test_split

# Define paths
dataset_dir = 'mendeley_lbc'
train_dir = f'{dataset_dir}_train'
val_dir = f'{dataset_dir}_val'
test_dir = f'{dataset_dir}_test'

# Create directories if they do not exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Create class directories
classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
for cls in classes:
    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(test_dir, cls), exist_ok=True)

# Split files into train, val, and test sets
for cls in classes:
    class_path = os.path.join(dataset_dir, cls)
    files = os.listdir(class_path)
    
    train_files, temp_files = train_test_split(files, test_size=0.3, random_state=42) # Train 70% of files
    val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)  # Split remaining 30% into 15% val and 15% test

    # Copy files to train, val, and test directories
    for f in train_files:
        shutil.copy(os.path.join(class_path, f), os.path.join(train_dir, cls, f))
    for f in val_files:
        shutil.copy(os.path.join(class_path, f), os.path.join(val_dir, cls, f))
    for f in test_files:
        shutil.copy(os.path.join(class_path, f), os.path.join(test_dir, cls, f))

print("Dataset has been successfully split and copied into train, validation, and test sets.")
