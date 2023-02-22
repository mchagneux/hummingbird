import torch
import numpy as np
import pandas as pd
import os
from torchvision.io.image import read_image

class HummingbirdDataset(torch.utils.data.Dataset):
  def __init__(self, root, data_type = "train"): # split is either train or test
    self.root = root
    self.annotations = pd.read_csv(os.path.join(root, data_type+'.csv'), 
                                   header = None, 
                                   names = ['img', 'xmin', 'ymin', 'xmax', 'ymax', 'cls'])
    class_names = pd.read_csv(os.path.join(root, 'class.csv'), header=None, names=['Name','ID'])
    self.class_names = class_names
    self.class_name_to_class_id = {name:id for name,id in zip(class_names.iloc[:,0], class_names.iloc[:,1])}
    # self.imgs = [os.path.join(data_type, file) for file in os.listdir(os.path.join(root, data_type))]
    self.imgs = self.annotations.iloc[:, 0]
  def __getitem__(self, idx):
    img_path = self.imgs[idx]
    image_annotations = self.annotations[self.annotations["img"] == img_path]
    n_rows = image_annotations.shape[0]
    boxes = []
    labels = []
    if n_rows == 1:
      img_path, xmin, ymin, xmax, ymax, cls_name = self.annotations.iloc[idx, :]
      if isinstance(cls_name, str):
        box = [xmin, ymin, xmax, ymax]
        label  = self.class_name_to_class_id[cls_name]
        labels.insert(0, label)
        boxes.insert(0, box)
    elif n_rows > 1:
      for i in np.arange(n_rows):
        img_path, xmin, ymin, xmax, ymax, cls_name = image_annotations.iloc[i, :]
        box = [xmin, ymin, xmax, ymax]
        label  = self.class_name_to_class_id[cls_name]
        labels.insert(i, label)
        boxes.insert(i, box)
    else:
      print("Oups, " + img_path + " is not in the annotations file")
    img = read_image(img_path) / 255
    target = {'boxes': torch.as_tensor(boxes), 
              'labels': torch.as_tensor(labels)}
    return img, target

  def __len__(self):
    return len(self.imgs)
