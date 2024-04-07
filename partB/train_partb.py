import torch
import wandb
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset, Subset
from tqdm.notebook import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('-wp', '--wandb_project', default="DL_Assignment_2", required=False, metavar="", type=str, help='Project name used to track experiments in Weights & Biases dashboard')
parser.add_argument('-we', '--wandb_entity', default="cs23m009", required=False, metavar="", type=str, help='Wandb Entity used to track experiments in the Weights & Biases dashboard')
parser.add_argument('-e', '--epochs', default=5, required=False, metavar="", type=int, help='Number of epochs to train model')
parser.add_argument('-b', '--batch_size', default=32, required=False, metavar="", type=int, help='Batch size used to train model')
parser.add_argument('-f', '--finetune_startegy', default="feature_extraction", required=False, metavar="", type=str, choices=["feature_extraction", "fine_tuning_all", "layer_wise_fine_tuning"], help='Different Fine tuning strategies')
parser.add_argument('-lr', '--learning_rate', default=0.0001, required=False, metavar="", type=float, help='Learning rate used to optimize model parameters')
args = parser.parse_args()

# Enter your wandb login key
wandb.login(key='')
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


'''
Function to prepare dataloader for train, test and val
Requires batch size and transform as input params
It splits train data into val data by 80 : 20 ratio
'''
def prepare_dataset(batch_size, transform):
    # Define the directory containing your dataset
    train_dir = 'inaturalist_12K/train'
    test_dir = 'inaturalist_12K/val'


    train_dataset = ImageFolder(train_dir, transform=transform)
    test_dataset = ImageFolder(test_dir, transform=transform)

    validation_ratio = 0.2
    class_labels = [label for _, label in train_dataset]

    class_id = defaultdict(list)
    for idx, label in enumerate(class_labels):
        class_id[label].append(idx)

    train_indices = []
    val_indices = []

    for y, ids in class_id.items():
        num_samples = len(ids)
        val_samples = int(validation_ratio * num_samples)
        np.random.shuffle(ids)  # Shuffle indices for random selection
        train_indices.extend(ids[val_samples:])
        val_indices.extend(ids[:val_samples])

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    
    val_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=val_sampler)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader, val_loader, test_loader

''' 
This function is used to train model the model per epoch basis
Input Params : 
model => pre-trained model
train_loader => training dataset 
loss_func => loss function (Cross Entropy)
optimizer => optimization function (Adam)
epoch => Epoch No.
'''

def train_per_epoch(model, train_loader, loss_func, optimizer, epoch):
    model.train()  # Set the model to training mode
    
    train_loss = 0.0
    correct_ans = 0
    num_samples = 0
        
    for data in tqdm(train_loader):
        inputs, labels = data[0].to(device), data[1].to(device)
            
        optimizer.zero_grad()
            
        outputs = model(inputs)
        loss = loss_func(outputs, labels)
            
        loss.backward()
        optimizer.step()
            
        train_loss += loss.item() * inputs.size(0)
        z, predicted = torch.max(outputs, 1)
        correct_ans += (predicted == labels).sum().item()
        num_samples += labels.size(0)
            
        
    epoch_loss = train_loss / num_samples
    epoch_accuracy = correct_ans / num_samples
    return epoch_loss, epoch_accuracy


''' 
This function is used to validate model the model per epoch basis
Input Params : 
model => pre-trained model
val_loader => validation dataset 
loss_func => loss function (Cross Entropy)
optimizer => optimization function (Adam)
epoch => Epoch No.
'''

def val_per_epoch(model, val_loader, loss_func, optimizer, epoch):
    model.eval()
    
    correct_ans = 0
    num_samples = 0
    val_loss = 0.0
    
    with torch.no_grad():
        for data in tqdm(val_loader):
            inputs, labels = data[0].to(device), data[1].to(device)

            outputs = model(inputs)
            loss = loss_func(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            z, predicted = torch.max(outputs, 1)
            correct_ans += (predicted == labels).sum().item()
            num_samples += labels.size(0)
            
    epoch_val_loss = val_loss / num_samples
    epoch_val_accuracy = correct_ans / num_samples
    return epoch_val_loss, epoch_val_accuracy


# Function to train the model
def train_model(model, train_loader, val_loader, loss_func, optimizer, num_epochs=5):
    for epoch in range(num_epochs):
        train_loss, train_acc = train_per_epoch(model, train_loader, loss_func, optimizer, epoch)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}')
        val_loss, val_acc = val_per_epoch(model, val_loader, loss_func, optimizer, epoch)
        print(f'Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')
        wandb.log({'train_loss': train_loss, 'train_accuracy': train_acc, 'val_loss' : val_loss, 'val_accuracy' : val_acc, 'epochs' : epoch + 1})
        

# Function to test the model
def test_model(model, test_loader, loss_func, optimizer):
    model.eval()
    
    correct_ans = 0
    num_samples = 0
    val_loss = 0.0
    
    with torch.no_grad():
        for data in tqdm(test_loader):
            inputs, labels = data[0].to(device), data[1].to(device)

            outputs = model(inputs)
            loss = loss_func(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            z, predicted = torch.max(outputs, 1)
            correct_ans += (predicted == labels).sum().item()
            num_samples += labels.size(0)
            
    test_loss = val_loss / num_samples
    test_acc = correct_ans / num_samples
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')

# Fine-tuning function
def fine_tune(model, strategy='feature_extraction'):
    if strategy == 'feature_extraction':
        # Freeze all layers except the final classification layer
        for name, param in model.named_parameters():
            if "fc" not in name:  # Skip parameters of the final fully connected layer
                param.requires_grad = False
    elif strategy == 'fine_tuning_all':
        # Unfreeze all layers and train the entire model
        for param in model.parameters():
            param.requires_grad = True
    elif strategy == 'layer_wise_fine_tuning':
        # Unfreeze and fine-tune only a subset of layers (e.g., only top layers)
        for i, param in enumerate(model.parameters()):
            if i < 6:  # Freeze the first 6 layers (adjust as needed)
                param.requires_grad = False
                
    return model

def main():
    weights= models.ResNet101_Weights.DEFAULT
    auto_transforms = weights.transforms()

    #Prepare Data
    train_loader, val_loader, test_loader = prepare_dataset(args.batch_size, auto_transforms)
    wandb.init(project = args.wandb_project, entity = args.wandb_entity)
    wandb.run.name = "ResNet"
    # Load pre-trained model
    model = models.resnet101(weights = weights)

    # Modify the final classification head
    num_classes = 10  # Number of classes in iNaturalist dataset
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    # Define loss function and optimizer
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    model = fine_tune(model, strategy=args.finetune_startegy)
    model.to(device)

    train_model(model, train_loader, val_loader, loss_func, optimizer, num_epochs=args.epochs)
    wandb.finish()
    

        
main()