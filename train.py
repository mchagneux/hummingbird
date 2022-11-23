import torch
import torchvision 
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torchvision.models.detection import retinanet_resnet50_fpn_v2
from torch.utils.data import DataLoader
import os 
import pandas as pd 
from PIL import Image
import math
import numpy as np 


def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

class HummingBirdDataset(torchvision.datasets.VisionDataset):

    def __init__(self, root, split):

        super().__init__(root, transform=get_transform(split == 'train'))

        self.annotations = pd.read_csv(os.path.join(root, split+'.csv'))

        class_names = pd.read_csv(os.path.join(root, 'class.csv'), header=None, names=['Name','ID'])
        self.class_name_to_class_id = {name:id for name,id in zip(class_names.iloc[:,0], class_names.iloc[:,1])}



    def __getitem__(self, index):

        image_path = self.annotations.iloc[index,0]
        xmin = self.annotations.iloc[index,1]
        ymin = self.annotations.iloc[index,2]
        xmax = self.annotations.iloc[index,3]
        ymax = self.annotations.iloc[index,4]

        cls_name = self.annotations.iloc[index,5]

        # TODO: correct to output no class and bounding box if there is nothing in the image
        if not isinstance(cls_name, str):
            boxes = []
            labels = []
        else: 
            box = [xmin, ymin, xmax, ymax]
            label  = self.class_name_to_class_id[cls_name]
            labels = [label]
            boxes = [box]


        image = Image.open(os.path.join(self.root,image_path))

        # the target is a dict with info on the object: box and position 
        target = {}
        target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
        target['labels'] = torch.as_tensor(labels, dtype=torch.int64)

        return self.transforms(image, target)
    
    def __len__(self):
        return len(self.annotations)

dataset = HummingBirdDataset(root='hummingbirds_dataset', split='train')
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)


model = retinanet_resnet50_fpn_v2(num_classes=5)
model.eval()


image, target = next(iter(dataloader))

prediction = model(image)
test = 0 