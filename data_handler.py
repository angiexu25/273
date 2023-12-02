import os
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
import torch

'''
anger = 0
contempt = 1
disgust = 2
fear = 3
happiness = 4
neutral = 5
sadness = 6
surprise = 7
'''

label_mapping = {"anger": 0, "contempt": 1, "disgust": 2, "fear": 3, "happiness": 4, "neutral": 5, "sadness": 6, "surprise": 7} 

class FacialExpressionsDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_name = self.annotations.iloc[index, 1]  # Adjust column index if necessary
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        label = self.annotations.iloc[index, 2]  # Adjust column index if necessary
        label = label_mapping[label.lower()]  # Convert label string to integer
        label = torch.tensor(label, dtype=torch.long)# Convert label to a tensor of type long
        
        if self.transform:
            image = self.transform(image)

        return (image, label)
