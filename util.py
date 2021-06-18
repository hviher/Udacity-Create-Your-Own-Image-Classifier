# Imports modules
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter  #may be needed

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

def load_data(args):
    
    data_dir = args.dir    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(255),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], 
                                                                     [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle = True)
    validloader = torch.utils.data.DataLoader(validation_data, batch_size=64, shuffle = True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle = True)
    
    #class_index = train_data.class_to_idx
      
    return trainloader, validloader, testloader, train_data


def load_names(args):   #filename = 'cat_to_name.json'
    with open(args.cat_to_name, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name


def network_setup(args):
    # Use GPU if requested by the user
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")      
    #gpu = args.gpu
    
    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.gpu and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cpu")
                               
    arch = args.arch
    
    if arch == 'vgg16':
        model = models.vgg16(pretrained = True)
    elif arch == 'densenet121':
        model = models.densenet121(pretrained = True)
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # Define the new classifier for the model
    if arch == 'vgg16':
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(25088, 4096)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout(args.dropout)),
            ('fc2', nn.Linear(4096, 512)),
            ('relu', nn.ReLU()),   
            ('fc3', nn.Linear(512, 102)),    
            ('output', nn.LogSoftmax(dim=1))
            ]))
    elif arch == 'densenet121':
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(1024, 512)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout(args.dropout)),
            ('fc3', nn.Linear(512, 102)),    
            ('output', nn.LogSoftmax(dim=1))
            ]))
        
    # Update the classifier in the model
    model.classifier = classifier
    
    # Move the model to Cuda
    model = model.to(device)

    # Define the loss
    criterion = nn.NLLLoss()
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), args.learning_rate)

    return model, criterion, optimizer



def validate():
    # Do validation on the test set
    # Set model to evaluation mode
    model.eval()
    #Set parameters
    #model.to(device)
    test_loss = 0
    accuracy = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            output = model(images)
            #calculate the loss
            loss = criterion(output, labels)                  
            test_loss += loss.item()
                  
            # Calculate accuracy
            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            equality = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equality.type(torch.FloatTensor)).item()
        print(f"Test loss: {test_loss/len(testloader):.3f}.. "
            f"Test accuracy: {accuracy/len(testloader):.3f}")
    
    
    
def save_checkpoint(args, model, optimizer, train_data):
    model.class_to_idx = train_data.class_to_idx

    checkpoint = {'arch': args.arch,
                  'learning_rate': args.learning_rate,
                  'batch_size': 64,
                  'epochs': args.epochs,
                  'classifier': model.classifier,
                  'optimizer': optimizer.state_dict(),
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx,}

    #Saving it to a file
    torch.save(checkpoint, args.save_dir)


def load_checkpoint(args):
    checkpoint = torch.load(args.checkpoint)
    arch = checkpoint['arch']
    model = getattr(torchvision.models, checkpoint['arch'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    #model.input_size = checkpoint['input_size']
    #model.output_size = checkpoint['output_size']
    optimizer = checkpoint['optimizer']
    learning_rate = checkpoint['learning_rate']
    epochs = checkpoint['epochs']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def process_image(args):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    ''' 
    # TODO: Process a PIL image for use in a PyTorch model
    image = Image.open(args.imagepth)
    
    # To resize the images keeping the original aspect ratio, we need to determine the length of the longest side 
    # of the image in order to determine the size tuple (width, height) to input into the thumbnail function, so that 
    # the shortest side will be 256, since the thumbnail function uses the smallest value for its calculation
    # size = [w,h]
    if image.size[0] > image.size[1]:
        size = [image.size[0], 256]
    else:
        size = [256, image.size[1]]
    
    # Resize the image and use ANTIALIAS to reduce the distortion
    image.thumbnail(size, Image.ANTIALIAS)
    
    # Crop the image to 224 x 224 by adjusting the cartesian coordinates needed for the crop function
    left = (256 - 224)/2
    upper = (256 - 224)/2
    right = (256 + 224)/2
    lower = (256 + 224)/2

    image = image.crop((left, upper, right, lower))
    
    # Adjust the color channel values (0-255) to floats between 0 and 1
    image = np.array(image)
    image = image/255
                       
    # Normalize the image as was done in the model above    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
                       
    image = ((image - mean) / std)
    
    #Move the color channel to be the first dimension from the third dimension (leaving the other two in order)
    image = np.transpose(image, (2, 0, 1))
    
    return image


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax







    
    

