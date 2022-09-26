#!/usr/bin/env python3
""" 
Author : Beng Hwee Toh

The Predict class which predict a flower image with a trained model

Example usage :
python predict.py --device=gpu --checkpoint=checkpoint.pth --image_path='flowers/valid/10/image_07102.jpg' --topk=5 --class_cat_path=cat_to_name.json

More sample :
python predict.py --device=cpu --checkpoint=checkpoint.pth --image_path='flowers/valid/5/image_05188.jpg' --topk=6 --class_cat_path=cat_to_name.json 


The example also is the default value. change the value according to the needs.

The model will load the checkpoint model which has been trained. Please call the train.py before use predict.
"""

# python bultin lib
import json
from collections import OrderedDict
from PIL import Image
# numpy
import numpy as np
# pytorch
import torch
import torch.nn.functional as F
import torchvision
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.autograd import Variable
# misc
from tqdm import tqdm
from classifier_base import ClassifierBase
import os
import input_args as args

class Predict(ClassifierBase):

    def __init__(self):
        ClassifierBase.__init__(self)
        # get input argument
        _args = args.predict_args()
        # var from input argument
        self.topk = _args.topk
        self.__device = _args.device
        self.image_path = _args.image_path
        self.checkpoint = _args.checkpoint
        self.class_cat_path = _args.class_cat_path

        self._init_labels(self.class_cat_path)
        self._load_checkpoint(self.checkpoint)

    def predict(self):
        self.__init_device();
        top_prob, top_classes = self.__predict(self.image_path , self.topk)
        class_label = top_classes[0]
        prob = top_prob[0]
        print("Using checkpoint file : ", self.checkpoint)
        print(f'\nPrediction\n---------------------------------')

        print(f'Flower      : {self.cat_to_name[class_label]}')
        print(f'Class       : {class_label}')
        print(f'Probability : {prob*100:.2f}%')
        print(f'\nTop probability result\n---------------------------------')
        for i in range(len(top_prob)):
            print(f"{self.cat_to_name[top_classes[i]]:<25} {top_prob[i]*100:.2f}%")
        return self.__predict(self.image_path , self.topk)

    def __predict(self, image_path , topk=5):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        ''' 
        # Implement the code to predict the class from an image file
        self.model.eval()
        
        # load image as torch.Tensor
        image = self.__process_image(image_path)
        
        # Unsqueeze returns a new tensor with a dimension of size one
        image = image.unsqueeze(0)
        
        # need to change to cuda
        if self.__device == "gpu":
            image = image.cuda()
        
        # Disabling gradient calculation 
        with torch.no_grad():
            
            output = self.model.forward(image)
          
            top_prob, top_labels = torch.topk(output, topk)
            top_prob = top_prob.exp()
            
        class_to_idx_inv = {self.model.class_to_idx[k]: k for k in self.model.class_to_idx}
        mapped_classes = list()
        
        # need to convert back to cpu model for numpy
        # Else will have an error
        if self.__device == "gpu":
            top_labels = top_labels.cpu()
            top_prob = top_prob.cpu()
        
        for label in top_labels.numpy()[0]:
            mapped_classes.append(class_to_idx_inv[label])
            
        return top_prob.numpy()[0], mapped_classes

    def __init_device(self):
        '''
        Inital device
        '''
        # Define your execution device
        if self.__device == "gpu":
            #if gpu not avaliable else fall back to cpu
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            else:
                print("Fail to detect gpu, fall back to cpu")
                self.device = torch.device("cpu")
                # need to set to cpu, need to check in the training method
                self.__device = "cpu"
        else:
            self.device = torch.device("cpu")

        # Convert model parameters and buffers to CPU or Cuda
        self.model.to(self.device) 
        print("The model will be running on", self.device, "device")

    def __process_image(self, image):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns an Numpy array
        '''       
        pil_image = Image.open(image).convert("RGB")
        
        # Any reason not to let transforms do all the work here?
        in_transforms = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(self._IMAGE_SIZE),
                                            transforms.ToTensor(),
                                            transforms.Normalize(self._MEANS, self._STD_DEV)])
        pil_image = in_transforms(pil_image)

        return pil_image

if __name__ == "__main__":
    Predict().predict();