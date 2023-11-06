import os
import sys
import random
import warnings
import csv
import torch
from PIL import Image
from torch import Tensor
import cv2
# from google.colab.patches import cv2_imshow
import numpy as np

from albumentations import Compose, Resize, RandomBrightnessContrast, RandomGamma
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss
from torch import nn

# Set some parameters
IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 21

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed

class Unet(nn.Module):
  def __init__(self,encoder,weight):
    super(Unet,self).__init__()

    self.backbone = smp.Unet(
        encoder_name = encoder,
        encoder_weights = weight,
        in_channels= 3,
        classes =1,
        activation = None
    )

  def forward(self, images, masks = None):
     logits = self.backbone(images)

     if masks != None:
       return logits, DiceLoss(mode = 'binary')(logits, masks) + nn.BCEWithLogitsLoss()(logits, masks)

     return logits

def preprocess(pil_image: Image):
    # Preprocessing data
    image_np = np.array(pil_image)
    image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)  # (h,w,c)
    transform = Compose([Resize(512, 512),ToTensorV2()])

    transformed_image = transform(image=image)['image']
    transformed_image = transformed_image.float()
    return transformed_image

def resnet50unet_model(input: Tensor, model_path: str):
    # Load the model
    model = Unet('resnet50','imagenet')
    model.load_state_dict(torch.load(model_path))

    # Set the model to evaluation mode (important if it has dropout or batch normalization layers)
    # model.eval()

    input_tensor = preprocess(input)

    logits_mask = model(input_tensor.to().unsqueeze(0)) # (c,h,w) -> (b, c, h, w)
    pred_mask = torch.sigmoid(logits_mask)
    
    pred_mask = (pred_mask > 0.5)*255.0
    
    mask_tensor = pred_mask[0]

    # Overlay the mask onto the image
    overlayed_image = input_tensor/255 - (input_tensor/255 *(1-mask_tensor) )
    return overlayed_image

def vgg16unet_model(input: Tensor, model_path: str):
    # Load the model
    model = Unet('vgg16','imagenet')
    model.load_state_dict(torch.load(model_path))

    # Set the model to evaluation mode (important if it has dropout or batch normalization layers)
    # model.eval()

    input_tensor = preprocess(input)

    logits_mask = model(input_tensor.to().unsqueeze(0)) # (c,h,w) -> (b, c, h, w)
    pred_mask = torch.sigmoid(logits_mask)
    
    pred_mask = (pred_mask > 0.5)*255.0
    
    mask_tensor = pred_mask[0]

    # Overlay the mask onto the image
    overlayed_image = input_tensor/255 - (input_tensor/255 *(1-mask_tensor) )
    
    return overlayed_image