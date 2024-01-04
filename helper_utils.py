from helper_datasets import CustomDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import time
import torch.nn as nn
import numpy as np
from random import shuffle
from sklearn.metrics import roc_curve, auc
import torch

DATASET_PATH = r'/home/jovyan/work/Project/USL-AD-EVAL/data/GKD_prep_patches'
DATASET_PATH = r'/home/jovyan/work/Project/USL-AD-EVAL/data/combined'

class normalize(object):
    def __call__(self, img):
        """
        :param img: (PIL): Image 

        :return: channelwise normalized image
        """
        mean = img.mean(axis=[1, 2], keepdim = True)  
        std = img.std(axis=[1, 2], keepdim = True) + 1e-1
        img = (img - mean) / std

        return img

def load_data(data_class: str, size: int, params):
    
    
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((size[0], size[1])),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #normalize()
        #transforms.Normalize(mean=0.0168, std=0.0112) # for gkd
        transforms.Normalize(mean=0.0524, std=0.0284)# for gkd
        #transforms.Normalize(mean=0.7346, std=0.0906) #for kssd
        #transforms.Normalize(mean=0.6116, std=0.1687) #for kssd
    ])
    test_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((size[0], size[1])),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), 
        #transforms.Normalize(mean=0.0168, std=0.0112) # for gkd
        #transforms.Normalize(mean=0.485, std=0.229) # for gkd
        transforms.Normalize(mean=0.0524, std=0.0284)# for gkd
        #transforms.Normalize(mean=0.7346, std=0.0906) #for kssd
        #transforms.Normalize(mean=0.6116, std=0.1687) #for kssd
    ])
    
    train_dataLoader = DataLoader(
        dataset=CustomDataset(root_dir=DATASET_PATH, defect_name=data_class, mode="train", transform=train_transforms), batch_size = params.get('batch_size', 8), shuffle = params.get('shuffle', True), num_workers = params.get('num_workers', 8) 
    )
    valid_dataLoader = DataLoader(
        dataset=CustomDataset(root_dir=DATASET_PATH, defect_name=data_class, mode="valid", transform=train_transforms), batch_size = params.get('batch_size', 8), shuffle = params.get('shuffle', True), num_workers = params.get('num_workers', 8)
    )
    test_dataLoader = DataLoader(
        dataset=CustomDataset(root_dir=DATASET_PATH, defect_name=data_class, mode="test", transform=test_transforms),batch_size = params.get('batch_size', 8), shuffle = params.get('shuffle', True), num_workers = params.get('num_workers', 8)
    )
    
    return train_dataLoader, valid_dataLoader, test_dataLoader

