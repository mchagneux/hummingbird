# -*- coding: utf-8 -*-
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
from utils import get_model
from tqdm import tqdm
from HummingbirdDataset import HummingbirdDataset
train_writer = SummaryWriter(filename_suffix = "_train")
test_writer = SummaryWriter(filename_suffix = "_test")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_classes = pd.read_csv("class.csv", header = None).shape[0]

model = get_model(device, num_classes)

trained_params = [p for p in model.parameters() if p.requires_grad]
print(sum(p.nelement() for p in trained_params))

my_path = './'
full_train_set = HummingbirdDataset(root = my_path, device=device, data_type = "train")
full_test_set = HummingbirdDataset(root = my_path, device=device, data_type = "test")
def collate_fn(batch):
    return tuple(zip(*batch))

train_dataloader = DataLoader(full_train_set, batch_size = 16, shuffle=True,  collate_fn = collate_fn)#, num_workers=torch.cuda.is_available() + 1)
test_dataloader = DataLoader(full_test_set, batch_size = 16, shuffle=True,  collate_fn = collate_fn)#, num_workers=torch.cuda.is_available() + 1)

optimizer = torch.optim.Adam(trained_params)

model.train()

n_epochs = 5

train_loss = []
test_loss = []

# Initial test_loss
epoch_loss = 0
for images, targets in tqdm(test_dataloader):
    loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values())
    epoch_loss += losses
    optimizer.zero_grad()
test_loss.append(epoch_loss)
train_writer.add_scalar("Loss/test/", epoch_loss, -1)
for epoch in range(n_epochs):
  # Training epoch
  epoch_loss = 0
  for images, targets in tqdm(train_dataloader):
    loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values())
    epoch_loss += losses
    optimizer.zero_grad()
    losses.backward()
    optimizer.step()
  train_loss.append(epoch_loss)
  train_writer.add_scalar("Loss/train/", epoch_loss, epoch)
  # Testing epoch epoch
  epoch_loss = 0
  for images, targets in tqdm(test_dataloader):
    loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values())
    epoch_loss += losses
  test_loss.append(epoch_loss)
  test_writer.add_scalar("Loss/test/", epoch_loss, epoch)

torch.save(model.state_dict(), "trained_weights_3")
