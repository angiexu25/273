import torch
import os
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.data import random_split
from torch.utils.data import Subset

# Import other files
from data_handler import FacialExpressionsDataset
from model import EmotionCNN


# Path to the CSV file
csv_file_path = "C:\\Users\\Angie Xu\\Emotion_Detection\\facial_expressions\\data\\legend.csv"
#img_dir_path = "C:\\Users\\Angie Xu\\Emotion_Detection\\facial_expressions\\images"
img_dir_path = "C:\\Users\\Angie Xu\\Emotion_Detection\\facial_expressions\\images_small"

# Define transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4683], std=[0.1852])
])

# Initialize dataset
dataset = FacialExpressionsDataset(csv_file=csv_file_path, img_dir=img_dir_path, transform=transform)

# Filter out non-existing files
existing_files = []
for i in range(len(dataset.annotations)):
    img_path = os.path.join(img_dir_path, dataset.annotations.iloc[i, 1])
    if os.path.exists(img_path):
        existing_files.append(i)

# Create a new dataset only with existing files
filtered_dataset = Subset(dataset, existing_files)

# Now split this filtered dataset
train_size = int(len(filtered_dataset) * 0.8)
val_size = len(filtered_dataset) - train_size
train_dataset, val_dataset = random_split(filtered_dataset, [train_size, val_size])

# Now create DataLoaders for each dataset
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

'''---check images loaded in dataset---
import matplotlib.pyplot as plt

def show_images(dataset, num_images=10):
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    for i in range(num_images):
        image, label = dataset[i]
        ax = axes[i]
        image = image.permute(1, 2, 0)  # Change the order of dimensions to HxWxC
        ax.imshow(image.numpy())
        ax.set_title(f"Label: {label}")
        ax.axis('off')
    plt.show()

show_images(dataset)
'''

# Initialize the model
model = EmotionCNN(num_classes=7)  # 7 classes

# Trainer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 1
# Number of epochs is the number of times you go through the entire dataset
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    for images, labels in train_loader:  # Loop over batches of data
        optimizer.zero_grad()           # Clear gradients for the next training iteration
        outputs = model(images)         # Pass the batch through the network
        loss = criterion(outputs, labels)  # Compute the loss
        loss.backward()                 # Backpropagate the gradients
        optimizer.step()                # Update the network's parameters

    # Validation phase
    model.eval()  # Set the model to evaluation mode
    total_val_loss = 0
    correct_predictions = 0
    with torch.no_grad():
        for val_images, val_labels in val_loader:
            val_outputs = model(val_images)
            val_loss = criterion(val_outputs, val_labels)
            total_val_loss += val_loss.item()
            
            _, predicted = torch.max(val_outputs, 1)
            correct_predictions += (predicted == val_labels).sum().item()

    # Calculate average loss and accuracy over the validation set
    avg_val_loss = total_val_loss / len(val_loader)
    val_accuracy = correct_predictions / len(val_loader.dataset)

    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
# Save the trained model
torch.save(model.state_dict(), 'emotion_detection_model.pth')

