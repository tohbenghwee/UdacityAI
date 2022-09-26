#!/usr/bin/env python3
""" 
Author : Beng Hwee Toh
The Traing class which train a model to predict flower species
Currently support 2 torch model vgg16 and densenet161

Example usage :
python train.py --device=gpu --arch=vgg16 --epochs=5 --learn_rate=0.001 --hidden_units=4096 --checkpoint=checkpoint.pth

The example also is the default value. change the value according to the needs.

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
import input_args as args

class Train(ClassifierBase):

    def __init__(self):
        '''
        Follow the step in order to use the train method
        '''
        ClassifierBase.__init__(self)

         # get input argument
        _args = args.train_args()
        # var from input argument
        self.learn_rate = _args.learn_rate
        self.hidden_units = _args.hidden_units
        self.epochs = _args.epochs
        # ['vgg16', 'densenet161']
        self.arch = _args.arch
        self.checkpoint = _args.checkpoint
        self.class_cat_path = _args.class_cat_path
        self.__device = _args.device
        print("Training architecture : ", self.arch)
        print("Hidden Units : ", self.hidden_units)
        print("learning rate : ", self.learn_rate)
        print("epochs : ", self.epochs)

    def train(self):
        '''
        Follow the step in order to use the train method
        single underscore using protected method from base class
        '''
        self.__init_dataset()
        self._init_labels(self.class_cat_path)
        self.__init_model()
        self.__init_loss_func_and_optimizer()
        self.__init_device()
        self.__train(self.epochs)
        self.__save_checkpoint(self.checkpoint)
        self._load_checkpoint(self.checkpoint)
        self.__validation()
       

    def __init_dataset(self):

         # Setup data tranforms
        self.data_transforms = {
            "train": transforms.Compose([transforms.RandomHorizontalFlip(p=0.30),
                                         transforms.RandomRotation(45),
                                         transforms.RandomResizedCrop(self._IMAGE_SIZE),
                                         transforms.ToTensor(),
                                         transforms.Normalize(self._MEANS, self._STD_DEV)]),
            "validate": transforms.Compose([transforms.Resize(self._IMAGE_SIZE + 1),
                                            transforms.CenterCrop(self._IMAGE_SIZE),
                                            transforms.ToTensor(),
                                            transforms.Normalize(self._MEANS, self._STD_DEV)]),
            "test": transforms.Compose([transforms.Resize(self._IMAGE_SIZE + 1),
                                        transforms.CenterCrop(self._IMAGE_SIZE),
                                        transforms.ToTensor(),
                                        transforms.Normalize(self._MEANS, self._STD_DEV)])
        }
        # Load the datasets with ImageFolder
        self.image_datasets = {
            "train": datasets.ImageFolder(self._TRAIN_DIR, transform=self.data_transforms["train"]),
            "validate": datasets.ImageFolder(self._VALIDATE_DIR, transform=self.data_transforms["validate"]),
            "test": datasets.ImageFolder(self._TEST_DIR, transform=self.data_transforms["test"])
        }
        # Using the image datasets and the trainforms, define the dataloaders
        self.dataloaders = {
            "train": torch.utils.data.DataLoader(self.image_datasets["train"], batch_size=self._BATCH_SIZE, shuffle=True),
            "validate": torch.utils.data.DataLoader(self.image_datasets["validate"], batch_size=self._BATCH_SIZE),
            "test": torch.utils.data.DataLoader(self.image_datasets["test"], batch_size=self._BATCH_SIZE)
        }


    def __init_model(self):
        '''
        Inital loss pretain model and attach new classifier
        '''
        # Choose vgg16 as the model
        output_size = len(self.cat_to_name)
        input_size = 0
        # Using ['vgg16', 'densenet161']
        if self.arch == "vgg16":
            self.model = models.vgg16(pretrained=True)
            input_size = self.model.classifier[0].in_features
        else:
            self.model = models.densenet161(pretrained=True)
            input_size = self.model.classifier.in_features

        # Prevent backpropagation on parameters
        for param in self.model.parameters():
            param.requires_grad = False 
    
        feature = self.hidden_units
        # Define own classifier
        new_classifier = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(in_features= input_size, out_features= feature , bias=True)),
                ('relu1', nn.ReLU()),
                ('dropout', nn.Dropout(p=0.5)),
                ('fc2', nn.Linear(in_features= feature, out_features= feature , bias=True)),
                ('relu2', nn.ReLU()),
                ('dropout', nn.Dropout(p=0.5)),
                ('output2', nn.Linear(in_features= feature, out_features= output_size, bias=True)),
                # LogSoftmax is needed by NLLLoss criterion
                ('softmax', nn.LogSoftmax(dim=1))
            ]))

        # Set the original classifier with our own
        self.model.classifier = new_classifier

    def __init_loss_func_and_optimizer(self):
        '''
        Inital loss function and optimizer
        '''
        # The negative log likelihood loss as criterion.
        self.loss_fn = nn.NLLLoss()
        # Adam: Stochastic Optimization
        self.optimizer = optim.Adam(self.model.classifier.parameters(), lr=self.learn_rate)

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
        
    def __check_accuracy(self):
        self.model.eval()
        accuracy = 0.0
        total = 0.0
    
        with torch.no_grad():
            for data in self.dataloaders['validate']:
                images, labels = data
                # add this line else it will throw error when using gpu
                # any better way to switch between cpu and gpu ?
                if self.__device == "gpu":
                    images, labels = images.cuda(), labels.cuda()
                # run the model on the test set to predict labels
                outputs = self.model(images)
                # the label with the highest energy will be our prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                accuracy += (predicted == labels).sum().item()
        
        # compute the accuracy over all test images
        accuracy = (100 * accuracy / total)
        return(accuracy)

    def __train(self, epochs):
        
        for epoch in range(epochs):
            running_loss = 0.0
            accuracy = 0
            for (images, labels) in tqdm(self.dataloaders['train']):
                # get the inputs
                images = Variable(images.to(self.device))
                labels = Variable(labels.to(self.device))

                # zero the parameter gradients
                self.optimizer.zero_grad()
                # predict classes using images from the training set
                outputs = self.model(images)
                # compute the loss based on model output and real labels
                loss = self.loss_fn(outputs, labels)
                # backpropagate the loss
                loss.backward()
                # adjust parameters based on the calculated gradients
                self.optimizer.step()
                running_loss += loss.item()     # extract the loss value

            # Compute and print the average accuracy fo this epoch when tested over all 10000 test images
            accuracy = self.__check_accuracy()
            print('epoch', epoch + 1,' accuracy over the whole validate set is %d %%' % (accuracy))
            print('[epoch %d] loss: %.3f' % (epoch + 1 , running_loss / len(self.dataloaders['train'])))
            

    def __validation(self):
        '''
        Do a validation on the test data set and evalation the overall accuracy
        '''
        correct = 0
        total = 0
        total_images = len(self.dataloaders["test"].batch_sampler) * self.dataloaders["test"].batch_size
        if self.__device == "gpu":
            self.model.cuda()
        # Disabling gradient calculation
        with torch.no_grad():
            for (images, labels) in tqdm(self.dataloaders["test"]):
                # Move images and labeles perferred device
                # if they are not already there
                images = images.to(self.device)
                labels = labels.to(self.device)
                # need to set this in cuda device
                if self.__device == "gpu":
                    images = images.cuda()
                    labels = labels.cuda()
            
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Accurately classified {(100 * correct // total):d}% of {total_images} images.')

    def __save_checkpoint(self, file='checkpoint.pth'):
        '''
        Save model to file
        '''
        self.model.class_to_idx = self.image_datasets['train'].class_to_idx
        model_state = {
            'arch': self.arch,
            'epoch': self.epochs,
            'state_dict': self.model.state_dict(),
            'optimizer_dict': self.optimizer.state_dict(),
            'classifier': self.model.classifier,
            'class_to_idx': self.model.class_to_idx,
        }
        print("Saving checkpoint to : " + self.checkpoint)
        torch.save(model_state, file)

if __name__ == "__main__":
    Train().train();