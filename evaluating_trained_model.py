# -*- coding: utf-8 -*-
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
from torch.utils.data import DataLoader
import math
from utils import get_model
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
num_classes = pd.read_csv("class.csv", header = None).shape[0]

model = get_model(device, num_classes)
# os.chdir("drive/MyDrive/Colab_Notebooks/00_training_detector_directory/")
# os.getcwd()

trained_params = [p for p in model.parameters() if p.requires_grad]
print(sum(p.nelement() for p in trained_params))

model.load_state_dict(torch.load("trained_weights",map_location = device))
model.eval()

class_names = pd.read_csv("class.csv", header = None)
colors = ["red", "blue", "green", "white", "purple", "orange"]

for test_file in os.listdir("test"):
  example_path = os.path.join("test", test_file)
  example_image = read_image(example_path)
  prediction = model([example_image / 255])
  predicted_scores = prediction[0]["scores"].detach().numpy()
  predicted_labels = np.array(prediction[0]["labels"])
  n_predictions = len(predicted_labels)
  score_threshold = 0.01
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
  dpi = 300  # Set the desired dpi value
  bbox_inches = 'tight'
  plt.savefig(os.path.join("prediction", test_file), dpi = dpi, bbox_inches=bbox_inches)
  plt.close()




