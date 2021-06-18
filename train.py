# PROGRAMMER: Heather Viher
# DATE CREATED: Jun 16, 2021                                  
# REVISED DATE: 
# PURPOSE: Basic usage: python train.py data_directory
#          Prints out training loss, validation loss, and validation accuracy as the network trains
#          Options:
#               Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
#               Choose architecture: python train.py data_dir --arch "vgg13"
#               Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
#               Use GPU for training: python train.py data_dir --gpu
#
#
# Use argparse Expected Call with <> indicating expected user input:
#     python train.py --dir <directory with images> --arch <model> --learning_rate <learning rate> --epochs <epochs> 
#                     --dropout<dropout rate> --save_dir <directory to save the model> --gpu
#             
# Example call:
#     python train.py --dir flowers --arch vgg16 --learning_rate 0.001 --epochs 3 --dropout 0.2 --save_dir checkpoint.pth --gpu
###


# Imports python modules
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import json
import argparse

from util import load_data, network_setup, save_checkpoint

def get_input_args():
    # Create Parser object named parser using ArgumentParser
    parser = argparse.ArgumentParser(description="Train a CNN model")
    
    # argument 1 - path to folder uncluding default
    parser.add_argument('--dir', type = str, default = 'flowers',
                        help = 'This is the path to the folder of input data')
    
    # arguement 2 - select CNN model architecture                                         
    parser.add_argument('--arch', dest='arch', type = str, default = 'vgg16', choices=['vgg16', 'densenet121'],
                        help = 'Select the CNN model architecture to use, VGG16 or DENSENET121')
       
    # arguement 3 - learning rate for training model                 
    parser.add_argument('--learning_rate', dest='learning_rate', type = float, default = 0.001,
                        help = 'This is the learning rate used to train the model')
    
    # arguement 4 - Number of epochs for training model                 
    parser.add_argument('--epochs', dest='epochs', type = int, default = 3,
                        help = 'This is the number of epochs used to train the model')  
    
    # arguement 5 - Dropout rate used in training the model                 
    parser.add_argument('--dropout', dest='dropout', type = float, default = 0.2,
                        help = 'This is the dropout rate used to train the model')      
    
    # arguement 6 - directory to save the checkpoint                 
    parser.add_argument('--save_dir', dest='save_dir', type = str, default = 'checkpoint_pth',
    #parser.add_argument('--save_dir', dest='save_dir', type = str, default = '../saved_models',
                        help = 'This is the directory to save the model (checkpoint) in')     
        
    # arguement 7 - train model on the GPU using CUDA                 
    parser.add_argument('--gpu', dest='gpu', default = False, action='store_true',
                        help = 'Add this option to train model on GPU using CUDA')      

    return parser.parse_args()

def train(args):
    # Use GPU if requested by the user
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")      
    #gpu = args.gpu
    
    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.gpu and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cpu")   

    epochs = args.epochs
    steps = 0
    print_every = 10

    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            steps += 1
        
            # Move input and label tensors to the default device (use cuda)
            images, labels = images.to(device), labels.to(device)
        
            # Zeros the gradients on each training pass
            optimizer.zero_grad()

            # Make a forward pass through the network to get the logits
            output = model.forward(images)
            # Use the logits to calculate the loss
            loss = criterion(output, labels)
            # Perform a backward pass through the network to calculate the gradients
            loss.backward()
            # Take a step with the optimizer to update the weights
            optimizer.step()
            # Calculate the training loss
            running_loss += loss.item()
      
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                # Track the loss and accuracy on the validation set
                with torch.no_grad():
                    for images, labels in validloader:
                        images, labels = images.to(device), labels.to(device)

                        output = model.forward(images)
                        batch_loss = criterion(output, labels)                  
                        valid_loss += batch_loss.item()
                  
                        # Calculate accuracy
                        ps = torch.exp(output)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                  
                print(f"Epoch {e+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Test loss: {valid_loss/len(validloader):.3f}.. "
                    f"Test accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()             

# Get input arguments
args = get_input_args()
#print(args)
                
#load and transform the data
trainloader, validloader, testloader, train_data = load_data(args)  

# Set up network with new classifier
model, criterion, optimizer = network_setup(args)

#train and save the model
train(args)

#save the model
save_checkpoint(args, model, optimizer, train_data)


    
    
    