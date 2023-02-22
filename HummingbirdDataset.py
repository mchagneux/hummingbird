import torch
import numpy as np
import pandas as pd
import os
from torchvision.io.image import read_image

class HummingbirdDataset(torch.utils.data.Dataset):
  def __init__(self, root, device, data_type = "train"): # split is either train or test
    self.root = root
    print('Building dataset...')
    self.device = device
    annotations = pd.read_csv(os.path.join(root, data_type+'.csv'), 
                                   header = None, 
                                   names = ['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'cls'])
    
    self.class_names = pd.read_csv(os.path.join(root, 'class.csv'), header=None, names=['Name','ID'])

    self.class_name_to_class_id = {name:id for name,id in zip(self.class_names.iloc[:,0], self.class_names.iloc[:,1])}

    self.filenames = list(np.unique(annotations.iloc[:, 0]))

    self.annotations_for_filename = dict()
    for filename in self.filenames: 
      self.annotations_for_filename[filename] = annotations[annotations['filename'] == filename]
    print('Dataset built.')

  def __getitem__(self, idx):
    filename = self.filenames[idx]
    img = read_image(filename) / 255.
    annotations_for_image = self.annotations_for_filename[filename]
    n_rows = annotations_for_image.shape[0]
    bboxes = []
    labels = []

    for idx in range(n_rows):
      _ , xmin, ymin, xmax, ymax, cls_name = annotations_for_image.iloc[idx,:]
      if isinstance(cls_name, str):
        bboxes.append([xmin, ymin, xmax, ymax])
        labels.append(self.class_name_to_class_id[cls_name])

    target =  {'boxes':torch.as_tensor(bboxes, device=self.device), 'labels':torch.as_tensor(labels, device=self.device)}

    return img.to(device=self.device), target  
  
  def __len__(self):
    return len(self.filenames)


