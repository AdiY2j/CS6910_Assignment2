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
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset, Subset
from tqdm.notebook import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-wp', '--wandb_project', default="DL_Assignment_2", required=False, metavar="", type=str, help='Project name used to track experiments in Weights & Biases dashboard')
parser.add_argument('-we', '--wandb_entity', default="cs23m009", required=False, metavar="", type=str, help='Wandb Entity used to track experiments in the Weights & Biases dashboard')
parser.add_argument('-n', '--num_filters', default=32, required=False, metavar="", type=int, choices= ['32', '64'], help='Number of filters') 
parser.add_argument('-e', '--epochs', default=5, required=False, metavar="", type=int, help='Number of epochs to train model')
parser.add_argument('-b', '--batch_size', default=32, required=False, metavar="", type=int, help='Batch size used to train model')
parser.add_argument('-bn', '--batch_norm', default=True, required=False, metavar="", type=bool, choices= ["True", "False"], help='Batch Normalization')
parser.add_argument('-da', '--data_aug', default=False, required=False, metavar="", type=bool, choices= ["True", "False"], help='Perform Data Augmentation')
parser.add_argument('-lr', '--learning_rate', default=0.0001, required=False, metavar="", type=float, help='Learning rate used to optimize model parameters')
parser.add_argument('-dp', '--dropout', default=0.2,  required=False, metavar="", type=float, help='Dropout Value')
parser.add_argument('-fs', '--filter_size', default=[2, 2, 2, 2, 2],  required=False, metavar="", type=list, help='Filter Size for 5 layers : [3, 3, 3, 3, 3], [2, 2, 2, 2, 2]')
parser.add_argument('-fo', '--filter_org', default="same", required=False, metavar="", type=str, choices=["same", "double", "half"], help='Filter Organization choices: ["same", "double", "half"]')
parser.add_argument('-a', '--activation', default="ReLU", required=False, metavar="", type=str, choices=['Mish', 'GELU', 'ReLU', 'SiLU'], help="Activation Function choices: ['Mish', 'GELU', 'ReLU', 'SiLU']")
args = parser.parse_args()

# Enter your wandb login key
wandb.login(key='72a114321dd97dbf11db7b15eb05b2660c2faa94')
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

'''
Function to prepare dataloader for train, test and val
Requires batch size and transform as input params
It splits train data into val data by 80 : 20 ratio
'''
def prepare_dataset(batch_size, data_aug):
    # Define the directory containing your dataset
    train_dir = 'inaturalist_12K/train'
    test_dir = 'inaturalist_12K/val'

    if data_aug :
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else :
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


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

# SpeciesCNN class having 5 layers of Conv2d, Activation and Maxpool along with dense fully connected layer and softmax as last layer for classification of 10 classes
class SpeciesCNN(nn.Module):
    def __init__(self, num_classes, num_filters=32, filter_size=[3, 3, 3, 3, 3], dense_neurons=512, activation = 'ReLU', batch_norm = False, dropout_val = 0.0, filter_org = 'double'):
        super(SpeciesCNN, self).__init__()
        act_func = nn.ReLU()
        match activation :
            case 'ReLU':
                act_func = nn.ReLU
            case 'GELU':
                act_func = nn.GELU
            case 'SiLU':
                act_func = nn.SiLU
            case 'Mish':
                act_func = nn.Mish
            case 'LeakyReLU':
                act_func = nn.LeakyReLU
            case 'Sigmoid':
                act_func = nn.Sigmoid

        match filter_org :
            case 'same':
                filters = [num_filters] * 5
            case 'double':
                filters = [num_filters * (2 ** i) for i in range(5)]
            case 'half':
                filters = [num_filters // (2 ** i) for i in range(5)]
                
                
                
        # Convolutional block 1
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=filters[0], kernel_size=filter_size[0], padding=0)
        self.act_1 = act_func()
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        if batch_norm :
            self.batch_1 = nn.BatchNorm2d(filters[0])
        
        # Convolutional block 2
        self.conv_2 = nn.Conv2d(in_channels=filters[0], out_channels=filters[1], kernel_size=filter_size[1], padding=0)
        self.act_2 = act_func()
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        if batch_norm :
            self.batch_2 = nn.BatchNorm2d(filters[1])
        
        # Convolutional block 3
        self.conv_3 = nn.Conv2d(in_channels=filters[1], out_channels=filters[2], kernel_size=filter_size[2], padding=0)
        self.act_3 = act_func()
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        if batch_norm :
            self.batch_3 = nn.BatchNorm2d(filters[2])
        
        # Convolutional block 4
        self.conv_4 = nn.Conv2d(in_channels=filters[2], out_channels=filters[3], kernel_size=filter_size[3], padding=0)
        self.act_4 = act_func()
        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        if batch_norm :
            self.batch_4 = nn.BatchNorm2d(filters[3])
        
        # Convolutional block 5
        self.conv_5 = nn.Conv2d(in_channels=filters[3], out_channels=filters[4], kernel_size=filter_size[4], padding=0)
        self.act_5 = act_func()
        self.pool_5 = nn.MaxPool2d(kernel_size=2, stride=2)
        if batch_norm :
            self.batch_5 = nn.BatchNorm2d(filters[4])
        

        self.flatten = nn.Flatten()
        
        # Dense layers
        self.fc1 = nn.LazyLinear(out_features=dense_neurons, bias=True, device=None, dtype=None)
        self.fc1_activation = act_func()
        if dropout_val > 0.0 :
            self.dropout1 = nn.Dropout(dropout_val)
        self.fc2 = nn.Linear(dense_neurons, num_classes)
        
    def forward(self, x):
        x = self.pool_1(self.act_1(self.conv_1((x))))
        if hasattr(self, 'batch_1'):
            x = self.batch_1(x)

        x = self.pool_2(self.act_2(self.conv_2((x))))
        if hasattr(self, 'batch_2'):
            x = self.batch_2(x)

        x = self.pool_3(self.act_3(self.conv_3((x))))
        if hasattr(self, 'batch_3'):
            x = self.batch_3(x)

        x = self.pool_4(self.act_4(self.conv_4((x))))
        if hasattr(self, 'batch_4'):
            x = self.batch_4(x)

        x = self.pool_5(self.act_5(self.conv_5((x))))
        if hasattr(self, 'batch_5'):
            x = self.batch_5(x)
            
        x = self.flatten(x)
        x = self.fc1_activation(self.fc1(x))
        if hasattr(self, 'dropout1'):
            x = self.dropout1(x)
        x = F.softmax(self.fc2(x), dim=1)
        
        return x


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
        
    return model

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
    

def main():
    wandb.init(project = args.wandb_project, entity = args.wandb_entity)
        
    #Prepare Data
    train_loader, val_loader, test_loader = prepare_dataset(args.batch_size, args.data_aug)
    # Define the model
    model = SpeciesCNN(num_classes=10, num_filters = args.num_filters, filter_size = args.filter_size, activation = args.activation, batch_norm = args.batch_norm, dropout_val=args.dropout, filter_org = args.filter_org)
    model.to(device)
    # Define loss function and optimizer
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    model = train_model(model, train_loader, val_loader, loss_func, optimizer, num_epochs=args.epochs)
        
    test_model(model, test_loader,loss_func, optimizer)
    wandb.finish()
    
        
main()