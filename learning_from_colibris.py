# -*- coding: utf-8 -*-
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from torch.utils.data import DataLoader
import math
import torchvision
from torchvision.io.image import read_image
from torchvision import transforms
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from torchvision.models.detection import RetinaNet_ResNet50_FPN_V2_Weights, retinanet_resnet50_fpn_v2
from torchvision.models.detection.retinanet import RetinaNetHead, RetinaNetClassificationHead
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# os.chdir("drive/MyDrive/Colab_Notebooks/00_training_detector_directory/")
# os.getcwd()

pretrained_weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT

# On peut ici régler les tailles des images et différents autres arguments
# La liste des arguments est sur la doc
# https://github.com/pytorch/vision/blob/main/torchvision/models/detection/retinanet.py
# Dans la partie __init__ de la classe RetinaNet (autour de la ligne 350)
model = retinanet_resnet50_fpn_v2(weights = pretrained_weights, min_size = 400, max_size = 1300, 
topk_candidates = 20, detections_per_img = 10, trainable_backbone_layers=0)


out_channels = model.head.classification_head.conv[0].out_channels
num_anchors = model.head.classification_head.num_anchors
num_classes = pd.read_csv("class.csv", header = None).shape[0]
model.head.classification_head.num_classes = num_classes
cls_logits = torch.nn.Conv2d(out_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
torch.nn.init.normal_(cls_logits.weight, std=0.01)  # as per pytorch code
torch.nn.init.constant_(cls_logits.bias, -math.log((1 - 0.01) / 0.01))  # as per pytorcch code
# assign cls head to model
model.head.classification_head.cls_logits = cls_logits

model.to(device)

list_names =[
# 'head.classification_head.conv.0.0.weight', 
# 'head.classification_head.conv.0.1.weight', 
# 'head.classification_head.conv.0.1.bias', 
# 'head.classification_head.conv.1.0.weight', 
# 'head.classification_head.conv.1.1.weight', 
# 'head.classification_head.conv.1.1.bias', 
# 'head.classification_head.conv.2.0.weight', 
# 'head.classification_head.conv.2.1.weight', 
# 'head.classification_head.conv.2.1.bias', 
'head.classification_head.conv.3.0.weight', 
'head.classification_head.conv.3.1.weight', 
'head.classification_head.conv.3.1.bias', 
'head.classification_head.cls_logits.weight', 
'head.classification_head.cls_logits.bias', 
# 'head.regression_head.conv.0.0.weight', 
# 'head.regression_head.conv.0.1.weight', 
# 'head.regression_head.conv.0.1.bias', 
# 'head.regression_head.conv.1.0.weight', 
# 'head.regression_head.conv.1.1.weight', 
# 'head.regression_head.conv.1.1.bias', 
# 'head.regression_head.conv.2.0.weight', 
# 'head.regression_head.conv.2.1.weight', 
# 'head.regression_head.conv.2.1.bias', 
'head.regression_head.conv.3.0.weight', 
'head.regression_head.conv.3.1.weight', 
'head.regression_head.conv.3.1.bias', 
'head.regression_head.bbox_reg.weight', 
'head.regression_head.bbox_reg.bias']
# Set requires_grad to false
# for n, p in model.named_parameters():
#      if n not in list_names:
#          p.requires_grad = False

trained_params = [p for p in model.parameters() if p.requires_grad]
print(sum(p.nelement() for p in trained_params))

from HummingbirdDataset import HummingbirdDataset
my_path = './'
full_train_set = HummingbirdDataset(root = my_path, device=device, data_type = "train")
# def my_collate(batch):
#   image = [item[0] for item in batch]
#   target = [item[1] for item in batch]
#   return [image, target]

def collate_fn(batch):
    return tuple(zip(*batch))


dataloader = DataLoader(full_train_set, batch_size = 32, shuffle=True,  collate_fn = collate_fn)#, num_workers=torch.cuda.is_available() + 1)

# optimizer = torch.optim.SGD(trained_params, lr=0.005,
#                             momentum=0.9, weight_decay=0.0005)
optimizer = torch.optim.Adam(trained_params)
                            # update the learning rate
 # and a learning rate scheduler
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
#                                              step_size=3,
#                                              gamma=0.1)

model.train()
from tqdm import tqdm

n_epochs = 200
# classification_losses = np.zeros((len(dataloader), n_epochs))
# bbox_losses =  np.zeros((len(dataloader), n_epochs))

my_loss = []
for epoch in range(n_epochs):
  n_batch = 0
  for images, targets in tqdm(dataloader):
    # images = list(image.to(device) for image in images)
    # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    loss_dict = model(images, targets)
    # classification_losses[epoch, n_batch] = loss_dict.get("classification").detach().numpy()
    # bbox_losses[epoch, n_batch] = loss_dict.get("bbox_regression").detach().numpy()
    losses = sum(loss for loss in loss_dict.values())
    optimizer.zero_grad()
    losses.backward()
    optimizer.step()
    n_batch = n_batch + 1
    print(losses)
  my_loss.append(losses)
#  lr_scheduler.step()

torch.save(model.state_dict(), "trained_weights_50epochs_collab_full_retrain")
