#!/usr/bin/env python3                                                                       
# PURPOSE: Create a function that retrieves the following 3 command line inputs 
import argparse
from unicodedata import decimal

def train_args():
    """
    For training script
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()

    # Model architecture
    parser.add_argument('--arch', type=str, default='vgg16', choices=['vgg16', 'densenet161'] , help="vgg16 or densenet161")

    # Model hyperparameters
    parser.add_argument('--learn_rate', type=float, default=0.001,help="Learning rate.")
    parser.add_argument('--hidden_units', type=int, default='4096',help="Number of hidden units.")
    parser.add_argument('--epochs', type=int, default='5',help="Training epochs for the model.")

    # device option
    parser.add_argument('--device', type=str, default='gpu', choices=['gpu', 'cpu'],help="Either gpu or cpu , fall back to cpu if gpu not detected")
    
    # checkpoint , only use when you need to save the checkpoint as a different name to prevent overwriting
    # Give full path if you need to store in other folder such as /opt/temp/checkpoint.pth
    parser.add_argument('--checkpoint', type=str, default='checkpoint.pth',help="The checkpoint file name or full path")
    
     # load category file
    parser.add_argument('--class_cat_path', type=str, default=r'cat_to_name.json',help="A JSON file path that maps the class values to other category names")
    
    return  parser.parse_args()



def predict_args():
    """
    For predict script
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()

    # device option
    parser.add_argument('--device', type=str, default='gpu', choices=['gpu', 'cpu'],help="Either gpu or cpu , fall back to cpu if gpu not detected")

    # image to predict and checkpoint , (extra parameter in case a new image / checkpoint is needed)
    parser.add_argument('--image_path', type=str, default=r"flowers/valid/10/image_07102.jpg", help=r"A image path which exist, if full path start with a /folder/flower.jpg")
    # Give full path if you need to store in other folder such as /opt/temp/checkpoint.pth
    parser.add_argument('--checkpoint', type=str, default='checkpoint.pth',help="The checkpoint file name or full path")

    # predict number of class
    parser.add_argument('--topk', type=int, default='5',help="print out the top K classes along with associated probabilities")

    # load category file
    parser.add_argument('--class_cat_path', type=str, default=r'cat_to_name.json',help="A JSON file path that maps the class values to other category names")

    return  parser.parse_args()