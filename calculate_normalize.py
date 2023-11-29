import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import os
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

"""
class FacialExpressionsDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 1])
        image = Image.open(img_path)
        label = self.annotations.iloc[index, 2]  # Assuming the label is in the third column
        if self.transform:
            image = self.transform(image)
        return (image, label)
"""

class FacialExpressionsDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 1])
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        label = self.annotations.iloc[index, 2]  # Assuming the label is in the third column
        if self.transform:
            image = self.transform(image)
        return (image, label)


csv_file_path = "C:\\Users\\Angie Xu\\Emotion_Detection\\facial_expressions\\data\\legend.csv"
img_dir_path = "C:\\Users\\Angie Xu\\Emotion_Detection\\facial_expressions\\images"

transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize all images to 256x256
    transforms.CenterCrop(256),     # Crop the images to 256x256
    transforms.ToTensor()           # Convert the image to a PyTorch tensor
])

# Assuming 'FacialExpressionsDataset' is your dataset class
dataset = FacialExpressionsDataset(csv_file=csv_file_path, img_dir=img_dir_path, 
                                   transform=transform)

loader = DataLoader(dataset, batch_size=64, shuffle=False)

mean = 0.
std = 0.
nb_samples = 0.

for data, _ in loader:
    batch_samples = data.size(0)
    data = data.view(batch_samples, data.size(1), -1)
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    nb_samples += batch_samples

mean /= nb_samples
std /= nb_samples

print("Mean: ", mean)
print("Std: ", std)
