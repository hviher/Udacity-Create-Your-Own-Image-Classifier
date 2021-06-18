# PROGRAMMER: Heather Viher
# DATE CREATED: Jun 17, 2021                                  
# REVISED DATE: 
# PURPOSE: Basic usage: python predict.py /path/to/image checkpoint
#          Options:
#               Return top K most likely classes: python predict.py input checkpoint --top_k 3
#               Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
#               Use GPU for inference: python predict.py input checkpoint --gpu
#
# Use argparse Expected Call with <> indicating expected user input:
#     python predict.py <checkpoint for saved model> <path to image to test> --topk <top number of predictions> --catnames <category to names json file> --gpu 
#             
# Example call:
#     python predict.py checkpoint.pth flowers/test/1/image_06743.jpg --topk 5 --catnames cat_to_name.json --gpu
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

from util import process_image, load_checkpoint, load_names

def get_input_args():
    # Create Parser object named parser using ArgumentParser
    parser = argparse.ArgumentParser(description="Train a CNN model")
       
    # arguement 1 - select saved model                                         
    parser.add_argument('checkpoint', type=str, default='checkpoint.pth',
                        help = 'Select the filepath of the saved model to load')
       
    # arguement 2 - select an image to test and predict the class                 
    parser.add_argument('imagepth', type = str, default = 'flowers/test/1/image_06743.jpg',
                        help = 'Enter the image filepath to classify')
    
    # arguement 3 - Top number of predictions to return                 
    parser.add_argument('--topk', dest='topk', type = int, default = 5,
                        help = 'This is the number of top predictions to return')    
                        
    # arguement 4 - select category to name index file                                         
    parser.add_argument('--cat_to_name', dest='cat_to_name', type=str, default='cat_to_name.json',
                        help = 'Select the filepah of the category to name index file')                        
        
    # arguement 5 - train model on the GPU using CUDA                 
    parser.add_argument('--gpu', dest='gpu', default = False, action='store_true',
                        help = 'Add this option to train model on GPU using CUDA')      

    return parser.parse_args()

#def predict(image_path, model, topk=5):
def predict(args, model, cat_to_name):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''    

    # Use GPU if requested by the user
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")          
    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.gpu and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cpu")                         
                        
    model.to(device)
    # Set model to evaluation mode
    model.eval()
        
    # Process the image for the model
    image = process_image(args)
    # Create a tensor from a numpy ndarray
    image = torch.from_numpy(np.array([image]))   
    # Change tensor to float    
    image = image.float()
    
    with torch.no_grad():
    # set the model to evaluation mode where the dropout probability is 0        
        output = model.forward(image.cuda())
    
    # Calculate the probabilities
    ps = F.softmax(output.data, dim =1)
    #ps = torch.exp(output).data
    
    # Get the top five probabilities
    top_ps, top_classes = ps.topk(args.topk, dim=1)    
    #top_p = np.array(ps.topk(topk)[0][0])      
    top_p = top_ps.tolist()[0]
    #index_to_class = {val:key for key, val in model.class_to_idx.items()}   #displays the numeric category
    index_to_class = {val:cat_to_name[k] for k, val in model.class_to_idx.items()}  #displays the category name

    top_class = [index_to_class[i] for i in top_classes.tolist()[0]]
    #top_class = [index_to_class[i] for i in np.array(ps.topk(topk)[1][0])]

    return top_p, top_class

                        
def main():
                            
# Prints out top k predictions with the probabilities
                        
    args = get_input_args()
    
    model = load_checkpoint(args)   
    cat_to_name = load_names(args) 
    
    probs, classes = predict(args, model, cat_to_name)
    
    print(f"Prediction results for image: {args.imagepth}")
    #print(probs)                                          
    #print(classes)                    
                        
    for c in range(len(classes)):
          print("#{}: {} - Probability: {:.2f}%".format((c+1), classes[c], probs[c]*100))
                
if __name__ == '__main__':
    main()



