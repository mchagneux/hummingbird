# -*- coding: utf-8 -*-
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
topk_candidates = 20, detections_per_img = 10)

out_channels = model.head.classification_head.conv[0].out_channels
num_anchors = model.head.classification_head.num_anchors
num_classes = pd.read_csv("class.csv", header = None).shape[0]
model.head.classification_head.num_classes = num_classes
cls_logits = torch.nn.Conv2d(out_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
torch.nn.init.normal_(cls_logits.weight, std=0.01)  # as per pytorch code
torch.nn.init.constant_(cls_logits.bias, -math.log((1 - 0.01) / 0.01))  # as per pytorcch code
# assign cls head to model
model.head.classification_head.cls_logits = cls_logits
if torch.cuda.is_available():
    model.cuda()
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
for n, p in model.named_parameters():
     if n not in list_names:
         p.requires_grad = False

trained_params = [p for p in model.parameters() if p.requires_grad]
sum(p.nelement() for p in trained_params)

model.load_state_dict(torch.load("trained_weights_50epochs_WE"))
model.eval()

class_names = pd.read_csv("class.csv", header = None)
colors = ["white", "black", "green", "red", "purple", "blue", "orange"]

example_path = "test/Session_4_Verrier_FF__cam_3__2022-07-26__08-00-00(1).JPG"
example_image = read_image(example_path)

prediction = model([example_image / 255])
predicted_scores = prediction[0]["scores"].detach().numpy()
predicted_labels = np.array(prediction[0]["labels"])
n_predictions = len(predicted_labels)
score_threshold = 0.2
kept_labels = [class_names.iloc[predicted_labels[j], 0] 
    for j in  range(n_predictions) if predicted_scores[j] > score_threshold]
kept_labels = [class_names.iloc[lab, 0] + " " + np.round(score * 100).astype("str") for lab, score in zip(predicted_labels, predicted_scores) if score > score_threshold]

kept_colors = [colors[lab] for lab, score in zip(predicted_labels, predicted_scores) if score > score_threshold]
kept_boxes = prediction[0]["boxes"][[j for j, score in zip(range(n_predictions), predicted_scores) if score > score_threshold]]
predicted_boxes = to_pil_image(draw_bounding_boxes(example_image,
                          boxes = kept_boxes,
                          labels = kept_labels,
                          colors = kept_colors,
                          width=2).detach())
fig, ax = plt.subplots()
ax.imshow(predicted_boxes)
plt.show()            


