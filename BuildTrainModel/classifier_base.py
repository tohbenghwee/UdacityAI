""" 
Author : Beng Hwee Toh

The base class for Train and Predict, 
trying to put common method and variable in base class

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
import input_args as args

class ClassifierBase(object):

    def __init__(self):
        ## define as commom value
        self._MEANS = [0.485, 0.456, 0.406]
        self._STD_DEV = [0.229, 0.224, 0.225]
        self._BATCH_SIZE = 32
        self._IMAGE_SIZE = 224
        self._DATA_DIR = 'flowers'
        self._TRAIN_DIR = self._DATA_DIR + '/train'
        self._VALIDATE_DIR = self._DATA_DIR + '/valid'
        self._TEST_DIR = self._DATA_DIR + '/test'

    def _init_labels(self , fileName = "cat_to_name.json"):
        '''
         Load the category labels into a dict 
        '''
        with open(fileName, 'r') as f:
            self.cat_to_name = json.load(f)
        print('Total Label Count : ' , len(self.cat_to_name))

    def _load_checkpoint(self, file='checkpoint.pth' ):
        '''
        Load model from file
        '''
        model_state = torch.load(file, map_location=lambda storage, loc: storage) 

        # load the correct arch 
        if  model_state['arch'] == "vgg16":
            self.model = models.vgg16(pretrained=True)
        else:
            self.model = models.densenet161(pretrained=True)

        self.model.classifier = model_state['classifier']
        self.model.load_state_dict(model_state['state_dict'])
        self.model.class_to_idx = model_state['class_to_idx']
        
    