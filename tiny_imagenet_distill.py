import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import time
import shutil
import requests, zipfile, io
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.models import resnet18, ResNet18_Weights

# Set you device to either GPU or CPU
# based on availability.
if torch.cuda.is_available(): 
    device = 'cuda'
else: 
    device = 'cpu'

# Download the tiny-imagenet-200 zip file and extract
# to a given directory which is current in this case.
imagenet_dir = os.path.join(".", "tiny-imagenet-200")
if not os.path.exists(imagenet_dir):
    tiny_imagenet_url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    r = requests.get(tiny_imagenet_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(".")

# Set the Directory and File Paths 
tiny_imagenet_path = "./tiny-imagenet-200"
train_dir = os.path.join(tiny_imagenet_path, "train")
val_dir = os.path.join(tiny_imagenet_path, "val")
val_annotations_file_path = os.path.join(val_dir, "val_annotations.txt")

# Change the directory structure and place the image files inside respective class
# folders in the val directory.
def reorg_tiny_imagenet_dir(val_dir, val_annotations_file_path):
    # create a directory for all the classes in tiny imagenet
    dir_class = os.path.join(val_dir, "class_dir")
    if not os.path.exists(dir_class):
        os.makedirs(dir_class)
    with open(val_annotations_file_path, 'r') as file:
        for line in file:
            items = line.strip().split("\t")
            image_file = items[0]
            image_class_name = items[1]
            image_dir = os.path.join(dir_class, image_class_name)
            if not os.path.exists(image_dir):
                os.makedirs(image_dir)
            val_image_dir = os.path.join(val_dir, "images")
            val_image_file = os.path.join(val_image_dir, image_file)
            if os.path.exists(val_image_file):
                try:
                    shutil.move(val_image_file, image_dir)
                except FileNotFoundError:
                    print("File not found to move")
                except PermissionError:
                    print("No permission to move the file")
                except Exception as err:
                    print("ERROR:", err)
    return dir_class

val_dir_reorg = reorg_tiny_imagenet_dir(val_dir, val_annotations_file_path)

# Transforms for Tiny ImageNet
transform_imagenet = transforms.Compose([
    # Resize the Image
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # ImageNet normalization
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Load Tiny ImageNet dataset
train_dataset = ImageFolder(root=train_dir, transform=transform_imagenet)
val_dataset = ImageFolder(root=val_dir_reorg, transform=transform_imagenet)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# Load and modify the teacher model for Tiny ImageNet
teacher_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
# We have to modify the final layer of the model 
# to output probability distribution for 200 class
teacher_model.fc = nn.Linear(teacher_model.fc.in_features, 200)
teacher_model = teacher_model.to(device)

# Initialize final layer weights
# Weights should neither be too small (which can lead to vanishing gradients) 
# nor be too large (which can lead to exploding gradients).
# Xavier initialization sets the weights so that the variance of activations 
# is roughly the same across all layers, ensuring stable training.
nn.init.xavier_uniform_(teacher_model.fc.weight)
teacher_model.fc.bias.data.fill_(0.01)

# Fine-tune the teacher model on Tiny ImageNet
def train_teacher_model(teacher_model, train_loader, num_epochs=10, learning_rate=0.0001):
    optimizer = optim.Adam(teacher_model.parameters(), lr=learning_rate)
    entr_loss = nn.CrossEntropyLoss()
    teacher_model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        weight_update_freq = 8
        for iteration, dataset in enumerate(train_loader, 1):
            images = dataset[0]
            labels = dataset[1]
            images, labels = images.to(device), labels.to(device)
            # We zero out the gradient to start fresh
            # and this is analogous to a variable being 
            # initialized to zero
            outputs = teacher_model(images)
            loss = entr_loss(outputs, labels)
            loss.backward()
            if iteration % weight_update_freq == 0:
                for w in teacher_model.parameters():
                    w.grad = w.grad / weight_update_freq
                optimizer.step()  # Update weights
                #optimizer.zero_grad()  # Clear accumulated gradients
            for w in teacher_model.parameters():
                w.grad.zero_()
            total_loss += loss.item()
        print(f"Teacher Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

# Knowledge distillation loss. KL divergence is the difference between the two 
# probability distributions
def know_distillation_loss(student_logits, teacher_logits, temperature=2.0):
    student_soft = nn.functional.log_softmax(student_logits / temperature, dim=1)
    teacher_soft = nn.functional.softmax(teacher_logits / temperature, dim=1)
    return nn.functional.kl_div(student_soft, teacher_soft, reduction='batchmean') * (temperature ** 2)

# Training student model with knowledge distillation
def train_student_model(student_model, teacher_model, train_loader, num_epochs=10, learning_rate=0.0001, temperature=2.0):
    optimizer = optim.Adam(student_model.parameters(), lr=learning_rate)
    teacher_model.eval()
    for epoch in range(num_epochs):
        student_model.train()
        total_loss = 0
        weight_update_freq = 8
        for iteration, dataset in enumerate(train_loader, 1):
            images = dataset[0]
            images = images.to(device)
            with torch.no_grad():
                teacher_outputs = teacher_model(images)
            student_outputs = student_model(images)
            # Here we don't use cross entropy loss but we call know_distillation_loss
            # function which finds the loss using KL divergence.
            loss = know_distillation_loss(student_outputs, teacher_outputs, temperature)
            optimizer.zero_grad()
            loss.backward()
            if iteration % weight_update_freq == 0:
                for w in student_model.parameters():
                    w.grad = w.grad / weight_update_freq
                # Update weights
                optimizer.step()
            for w in student_model.parameters():
                w.grad.zero_()
            total_loss += loss.item()
        print(f"Student Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

# Evaluate model
def evaluate_model(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for dataset in val_loader:
            images = dataset[0]
            labels = dataset[1]
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # Operate along the columns i.e from left to right
            _, predicted_class = torch.max(outputs, 1)
            for cl, lab in zip(predicted_class, labels):
                if cl.item() == lab.item():
                    correct += 1
            total += labels.size(0)
    accuracy = 100 * correct / total
    return accuracy

# Train and evaluate
print("Training Teacher Model on Tiny ImageNet...")
train_teacher_model(teacher_model, train_loader)

student_model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
student_model.classifier[1] = nn.Linear(student_model.last_channel, 200)  # Adjust for Tiny ImageNet
student_model = student_model.to(device)

print("Training Student Model via Knowledge Distillation on Tiny ImageNet...")
train_student_model(student_model, teacher_model, train_loader)

print("\nEvaluating Teacher Model...")
teacher_accuracy = evaluate_model(teacher_model, val_loader)
print(f"Teacher Model Accuracy: {teacher_accuracy:.2f}%")

print("\nEvaluating Student Model...")
student_accuracy = evaluate_model(student_model, val_loader)
print(f"Student Model Accuracy: {student_accuracy:.2f}%")

# Count parameters in teacher model
total_params = 0
for p in teacher_model.parameters():
    total_params += p.numel()

trainable_params = 0
for p in teacher_model.parameters():
    if p.requires_grad:
        trainable_params += p.numel()

print(f"Total parameters for the teacher model: {total_params}")
print(f"Trainable parameters for the teacher: {trainable_params}")

# Count parameters in student model
total_params = 0
for p in student_model.parameters():
    total_params += p.numel()

trainable_params = 0
for p in student_model.parameters():
    if p.requires_grad:
        trainable_params += p.numel()

print(f"Total parameters for the student model: {total_params}")
print(f"Trainable parameters for the student: {trainable_params}")
