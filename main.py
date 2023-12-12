import os
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data import Subset
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from matplotlib import image
import random
import time
# Import other files
from data_handler import FacialExpressionsDataset
from model import EmotionCNN
from flat_model import just_flat
from check_image import check_image_size
from weighted_loss import weighted_loss
from collections import Counter


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

#csv_file_path = os.path.join("C:/UCI/Project/facial_expressions/data/legend.csv")
#img_dir_path = os.path.join("C:/UCI/Project/facial_expressions/images")

print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

# Storing ID of current CUDA device
cuda_id = torch.cuda.current_device()
print(f"ID of current CUDA device: {torch.cuda.current_device()}")
print(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.empty_cache()

# Path to the CSV file
csv_file_path = "C:/UCI/Project/facial_expressions/data/legend.csv"
img_dir_path = "C:/UCI/Project/facial_expressions/images"
# img_dir_path = "C:\\Users\\Angie Xu\\Emotion_Detection\\facial_expressions\\images_small"

# Define transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4683], std=[0.1852])
])

# Initialize dataset
dataset = FacialExpressionsDataset(csv_file=csv_file_path, img_dir=img_dir_path, transform=transform)

# Filter out non-existing files
target_width = 350
target_height = 350
existing_files = []
bad_image = []
total_number_of_data = len(dataset.annotations)
for i in range(total_number_of_data):
    img_path = os.path.join(img_dir_path, dataset.annotations.iloc[i, 1])
    if os.path.exists(img_path) and check_image_size(img_path, target_width, target_height):
        existing_files.append(i)
    elif not check_image_size(img_path, target_width, target_height):
        bad_image.append(i)

# Create a new dataset only with existing files
filtered_dataset = Subset(dataset, existing_files)

# Now split this filtered dataset
train_dataset, val_dataset = random_split(filtered_dataset, [0.8, 0.2])

# Now create DataLoaders for each dataset
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=True)

# display one image to see the result
train_data_size = len(train_dataset)
one_random = random.randrange(0, train_data_size)
image_index = existing_files[one_random]
image_path = os.path.join(img_dir_path, dataset.annotations.iloc[image_index, 1])
example = image.imread(image_path)
plt.imshow(example, cmap='gray')
plt.xlabel("Expression: " + dataset.annotations.iloc[image_index, 2])
plt.show()
"""
train_data_size = len(train_dataset)
fig, ax = plt.subplots(1,5, figsize = (18, 6))
for i in range(5):
    one_random = random.randrange(0, train_data_size)
    image_index = existing_files[one_random]
    image_path = os.path.join(img_dir_path, dataset.annotations.iloc[image_index, 1])
    example = image.imread(image_path)
    ax[i].imshow(example, cmap='gray')
    ax[i].set_xlabel("Expression: " + dataset.annotations.iloc[image_index, 2])
plt.show()
"""

# show one bad image

bad_data_size = len(bad_image)
print(f"Number of bad images is {bad_data_size}")
one_random = random.randrange(0, bad_data_size)
image_index = bad_image[one_random]
image_path = os.path.join(img_dir_path, dataset.annotations.iloc[image_index, 1])
example = image.imread(image_path)
plt.imshow(example, cmap='gray')
plt.xlabel("Expression: " + dataset.annotations.iloc[image_index, 2])
plt.show()
"""
bad_data_size = len(bad_image)
print(f"Number of bad images is {bad_data_size}")
fig, ax = plt.subplots(1,5, figsize = (18, 6))
for i in range(5):
    one_random = random.randrange(0, bad_data_size)
    image_index = bad_image[one_random]
    image_path = os.path.join(img_dir_path, dataset.annotations.iloc[image_index, 1])
    example = image.imread(image_path)
    ax[i].imshow(example, cmap='gray')
    ax[i].set_xlabel("Expression: " + dataset.annotations.iloc[image_index, 2])
plt.show()
"""

# Initialize the model
model = EmotionCNN(num_classes=8)  # 8 classes
#model = just_flat(num_classes=8)

# Trainer
# define weighted loss
for i in range(total_number_of_data):
    dataset.annotations.iloc[i, 2] = dataset.annotations.iloc[i, 2].lower() # reduce uppercase emotion labels to lower case
emotion_dict = dict(Counter(dataset.annotations.iloc[:, 2]))

emotions = ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
num_of_sample = []
for emotion in emotions:
    num_of_sample.append(emotion_dict[emotion])

weighted = weighted_loss(total_number_of_data, num_of_sample, 8)
print(emotion_dict)
print(weighted)

weighted = torch.from_numpy(weighted) # convert np array to torch tensor
weighted = weighted.float()
weighted.to(device)
criterion = nn.CrossEntropyLoss(weight=weighted).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


start = time.time()
num_epochs = 5
# Number of epochs is the number of times you go through the entire dataset
for epoch in range(num_epochs):
    # Transfer model to GPU
    model.to(device)
    # Training phase
    model.train()  # Set the model to training mode，这句其实没有在train，只是set成train模式，下面的才是training
    
    for images, labels in train_loader:  # Loop over batches of data
        images, labels = images.to(device), labels.to(device)
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
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            val_outputs = model(val_images)
            val_loss = criterion(val_outputs, val_labels)
            total_val_loss += val_loss.item()
            
            _, predicted = torch.max(val_outputs, 1)
            '''
            This line determines the model's predicted labels for each image. 
            torch.max(val_outputs, 1) returns the indices of the maximum values along the dimension 1 (which represents class scores in your case). 
            The _ is a placeholder for the actual maximum values, which you don't need here.
            '''
            correct_predictions += (predicted == val_labels).sum().item()
            '''
            This line compares the predicted labels with the actual labels (val_labels) to count how many predictions were correct. 
            The result is added to correct_predictions.
            
            '''
    # Calculate average loss and accuracy over the validation set
    avg_val_loss = total_val_loss / len(val_loader)
    val_accuracy = correct_predictions / len(val_loader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
# Save the trained model
# torch.save(model.state_dict(), 'emotion_detection_model.pth')
spent = time.time()- start
print(f"Total time spent is {spent}s")