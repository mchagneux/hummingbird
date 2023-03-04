# -*- coding: utf-8 -*-
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from utils import get_model
from tqdm import tqdm
from HummingbirdDataset import HummingbirdDataset

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_classes = pd.read_csv("class.csv", header = None).shape[0]

model = get_model(device, num_classes)



trained_params = [p for p in model.parameters() if p.requires_grad]
print(sum(p.nelement() for p in trained_params))

my_path = './'
full_train_set = HummingbirdDataset(root = my_path, device=device, data_type = "train")

def collate_fn(batch):
    return tuple(zip(*batch))

dataloader = DataLoader(full_train_set, batch_size = 32, shuffle=True,  collate_fn = collate_fn)#, num_workers=torch.cuda.is_available() + 1)

optimizer = torch.optim.Adam(trained_params)

model.train()

n_epochs = 1

my_loss = []
for epoch in range(n_epochs):
  n_batch = 0
  for images, targets in tqdm(dataloader):
    loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values())
    optimizer.zero_grad()
    losses.backward()
    optimizer.step()
    n_batch = n_batch + 1
    print(losses)
  my_loss.append(losses)

torch.save(model.state_dict(), "trained_weights_200epochs_WE")
