import torch.utils.data as data
from PIL import Image
import os

class GetLoader(data.Dataset):
    def __init__(self, data_root, transform=None):
        self.root = data_root
        self.transform = transform

        # List to hold image paths and labels
        self.img_paths = []
        self.img_labels = []

        # Traverse the directory
        for label_idx, class_name in enumerate(os.listdir(self.root)):
            class_dir = os.path.join(self.root, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    self.img_paths.append(img_path)
                    self.img_labels.append(label_idx)

        self.n_data = len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        label = self.img_labels[index]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return self.n_data
