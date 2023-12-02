import torch.nn as nn


class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):  # Adjust num_classes based on your dataset
        super(EmotionCNN, self).__init__()
        
        # First block
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.25)
        
        # Second block
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.dropout2 = nn.Dropout(0.25)
        
        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=128 * 32 * 32, out_features=1024)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(in_features=1024, out_features=num_classes)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # First block
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Second block
        x = self.relu(self.conv3(x))
        x = self.pool2(x)
        x = self.relu(self.conv4(x))
        x = self.pool3(x)
        x = self.dropout2(x)
        
        # Fully connected layers
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)  # No softmax here as it's combined with nn.CrossEntropyLoss
        
        return x

   
    


