import torch
import math
from torchvision.models.detection import RetinaNet_ResNet50_FPN_V2_Weights, retinanet_resnet50_fpn_v2
from torchvision.models.detection.retinanet import RetinaNetHead, RetinaNetClassificationHead

def get_model(device, num_classes):
  pretrained_weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT

# On peut ici régler les tailles des images et différents autres arguments
# La liste des arguments est sur la doc
# https://github.com/pytorch/vision/blob/main/torchvision/models/detection/retinanet.py
# Dans la partie __init__ de la classe RetinaNet (autour de la ligne 350)
  model = retinanet_resnet50_fpn_v2(weights = pretrained_weights, 
  min_size = 128, max_size = 128, 
  topk_candidates = 5, detections_per_img = 5, trainable_backbone_layers=0)
  out_channels = model.head.classification_head.conv[0].out_channels
  num_anchors = model.head.classification_head.num_anchors
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
  'head.classification_head.conv.2.0.weight',
  'head.classification_head.conv.2.1.weight',
  'head.classification_head.conv.2.1.bias',
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
  'head.regression_head.conv.2.0.weight',
  'head.regression_head.conv.2.1.weight',
  'head.regression_head.conv.2.1.bias',
  'head.regression_head.conv.3.0.weight', 
  'head.regression_head.conv.3.1.weight', 
  'head.regression_head.conv.3.1.bias', 
  'head.regression_head.bbox_reg.weight', 
  'head.regression_head.bbox_reg.bias']
  # Set requires_grad to false
  for n, p in model.named_parameters():
       if n not in list_names:
           p.requires_grad = False
  return model;
