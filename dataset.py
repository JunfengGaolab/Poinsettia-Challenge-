import random
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# from model import CSRNet
preprocess = transforms.Compose([
    #transforms.Scale(256),
    transforms.RandomResizedCrop(size=224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])


def default_loader(path,train = True):
    img = Image.open(path).convert('RGB')

    label=[]
    if 'WFruit' in path:
        label.append(0.)
    else:
        label.append(1.)
        
    label=np.asarray(label) 
    
    img = preprocess(img)
    return img,torch.from_numpy(label)


class Pre_Trainset(Dataset):
    def __init__(self, Fruit_list,shape=None, transform=None, train=True, seen=0, batch_size=16, num_workers=4):
        random.shuffle(Fruit_list)
        
        self.fruit = Fruit_list
        self.nSamples = len(Fruit_list)
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __getitem__(self, index):
        fruit = self.fruit[index]
        img,label = default_loader(fruit)
        
        return img,label

    def __len__(self):
        return self.nSamples


    
    
    
    
    
    