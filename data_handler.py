import os
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset

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
        #if not os.path.exists(img_path):
        #    return None  # Return None if the image does not exist
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        label = self.annotations.iloc[index, 2]  # Adjust column index if necessary
        if self.transform:
            image = self.transform(image)
        return (image, label)
