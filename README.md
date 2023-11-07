

# Segment life saver: Breast cancer segmentation by ML models integrating Docker.

![](https://lh7-us.googleusercontent.com/yxm9IG7B7mdYPGwc9jOjBGTejOMcq7Igv0xLnnyYQoQj3puLtGBndIt9efRAs2GVTBXYUiEbKOvZPbPmCJFB3V1O3pjVyX0sGO1tYep8xp8cdmeFXvI3jh1jK4cxrKikloS2bM8UubO1ofgcHtw3qfw)

  
  

# About the challenge

Docker and Docker Hub are the starting point for practitioners to start their AI/ML journey and distribute their applications or models. We’re looking for hacks that use Docker products to help both beginners and advanced users get inspired, get started, and be productive within this exciting new frontier.

# Problem

Breast cancer is one of the most common types of cancer affecting millions of women worldwide, and it can also affect men, albeit less commonly. Awareness is crucial because early detection often leads to more effective treatment and a higher chance of survival.

Segmentation helps pathologists in making precise diagnoses by providing clear images of cancer cells, which is crucial for determining the stage of cancer. This process involves the identification and isolation of cancerous cells from non-cancerous cells and the surrounding tissue in images obtained from breast tissue biopsies. The goal is to accurately define the boundaries of the malignant cells to assess the progression and potential aggressiveness of the cancer.

![](https://lh7-us.googleusercontent.com/w9UEin5f_m91nIwlRGrB_Yn3F4t2uR9gEfeaAeyzxwYDmucy3dXSOVeIaNlpU3ZltOSwtxP6mYlkFk7RfthjY_74WmLFMzBNGrtHej6HGVuhjSUZm6aE7FyMWrA2jA6z4R3eiiWDAbcUUXmLJ-dNXz8)

  
  
  

# Methodology

  

## Backend
    

### Training Model

The objective of the model is to segment the tumors in breast cancer slide images. With the result of the model, we can notice the tumors more easily. To achieve this objective, we utilized cancer images and segmentation label from [HistomicsUI (kitware.com)](https://demo.kitware.com/histomicstk/histomicstk#?image=5bbdee62e629140048d01b0d&bounds=-42540%2C0%2C127990%2C84350%2C0) in which the dataset associated with the paper:  [Amgad M, Elfandy H, ..., Gutman DA, Cooper LAD. Structured crowdsourcing enables convolutional segmentation of histology images. Bioinformatics. 2019. doi: 10.1093/bioinformatics/btz083](https://academic.oup.com/bioinformatics/article/35/18/3461/5307750)

  

For the model training we have tried on 2 models in following

1.  #### ResNet50 + Unet
    

![](https://lh7-us.googleusercontent.com/wF6ZJlovt2czeEWSF64WmH3CJkcalI_pG2A-1_bNZrSbj8A9jMuXkPxe3bcWHqTb0G1ZaowcqXM5ZFkujKZfg-cWH5eM9i3xKUhHCeEDUdpPQn9A3tuTuyZ9CTiJXfEoQZY-OTbFJA0xswi3FtqrxKY)

source: [The Annotated ResNet-50. Explaining how ResNet-50 works and why… | by Suvaditya Mukherjee | Towards Data Science](https://towardsdatascience.com/the-annotated-resnet-50-a6c536034758)

  

![](https://lh7-us.googleusercontent.com/b593jBgzsjAVTwBz4kzs0ugCPENtyPgnBwKhfzhHwuGT_XTthRwZulFO3KM94CswT3QQLNeuehAVI2ZdD6NBgEor-a0Qsd3Vq2j_G0gfxNFS5lGkM4_bUkJwijd8I_AR3vBGiH86oYf6KwmM5QjxUAU)

Source: [U-Net: Convolutional Networks for Biomedical Image Segmentation (uni-freiburg.de)](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)

  

For the first model, ResNet50 is encoder and Unet is decoder.

ResNet50, short for "Residual Network with 50 layers," is a deep convolutional neural network (CNN) architecture that is a part of the ResNet family. ResNet was introduced to address the problem of vanishing gradients in very deep neural networks. ResNet50 consists of 50 layers and is known for its ability to train very deep networks effectively. It employs residual connections, allowing information to flow through the network more easily.  

U-Net is a specialized architecture for image segmentation tasks, especially in medical image analysis. It has an encoder-decoder structure, where the encoder captures context and features from the input image, and the decoder generates pixel-wise segmentation masks. U-Net is characterized by skip connections that help preserve fine-grained details during the upsampling process.

2.  #### VGG16 + Unet
    
 ![](https://lh7-us.googleusercontent.com/C7Rq0r4RF2Q851tLvJQYvyb2mXGw8ainP89fYZPOYEktitmcoCw3xBNcZc1exorUzYguR3an9FWn9qUJz7wVfsGllb3bEfR1Cmg4N6asxE66HR0qmqFzABC0nyyt1rc25mlYwMty2Nw7niJBH2-FC4M)

Src: [An overview of VGG16 and NiN models | by Khuyen Le | MLearning.ai | Medium  
](https://medium.com/mlearning-ai/an-overview-of-vgg16-and-nin-models-96e4bf398484)For Next model, VGG16 is encoder and Unet is decoder.

VGG16 is a convolutional neural network (CNN) architecture that was developed by the Visual Geometry Group (VGG) at the University of Oxford. It is known for its simplicity and effectiveness in image classification tasks. VGG16 consists of 16 weight layers, including 13 convolutional layers and 3 fully connected layers. It has been widely used as a feature extractor for image-related tasks, such as object detection and image segmentation.

U-Net is a specialized CNN architecture designed for image segmentation tasks, particularly in medical image analysis and related fields. U-Net is characterized by its U-shaped architecture, which consists of an encoder and a decoder. The encoder portion captures context and features from the input image, while the decoder part generates a pixel-wise segmentation mask. U-Net has skip connections that help preserve fine-grained details during the upsampling process.

  
  

  

### Fast API  
![](https://lh7-us.googleusercontent.com/W-CP-O8Oywf6znLPpM1We8YSBa51ugDsGsRI3uJhDt4SkuvXeAEyKFvGlKvQ9jONzJg87Ucz-XGUOMZ6EbAdhUyoJ7FprTFFRbDdPvPJh8UpA6Nmv8bVcT5RYtOZpVz7wk94za1OOYrx6k2E5hJuwdE)

For Fast API, we provided 2 api for segmented the model. Which we will get the input from the user, then it will pass to the model that we have trained in the previous part.

![](https://lh7-us.googleusercontent.com/j8t11BWSMZAR1xzmv9p559sQkCF_Ldu3JrwyJDl8igUfr98boBTUeSHv8TIww98oAm-yJGaYA7pVlu9zhYG9kSqs5tNAwrLtFdy5vsEhETRYwBONyLcmJB6WhQE1geAbrZDlZSIq6ZNbMPyjJ04J_JU)For resnet50_unet, users need to send the image to the api. Then the model will predict the area of tumors and send it back to the user as a png image.

![](https://lh7-us.googleusercontent.com/xJTWioq4kDLD3j3iKpLJCER3Hfq2pObIHVqo1uj1IcOmNHPWzMYPaX6DeJ2zlnBZqe2l2SufaR4RI38qhmwHdAJk1sS2Owy26U8tFpX_aYiC-ujXBeuVVFGpztfWT3iWXTFsHJHsoyITaVIkbkkzn_w)For vgg16_unet, users also need to send the image to the api. Then the model will predict the area of tumors and send it back to the user. It may take a little time to process.

## Frontend

![](https://lh7-us.googleusercontent.com/wDFEFSXHkKt91OAAm9FLD5WXKn1qOTuKc5DTIZ4mi2MqsH55aisBY7S1yMQkB5E3UG6q0L0E-mgta8Vtl4avggUOOLjg9fJghyhn6F3_Ti79ugwFduGiAqmYRNgG33V2hiBr6Cc_xoadZNBz2DrVrjE)  
We have developed the frontend using ReactJS to connect with FastAPI, so the user can try uploading the image to segment the tumor more easily.

## Docker

![](https://lh7-us.googleusercontent.com/tUBdQwhnzG159NHACtp3k8Ip7dLPRE28rqxzG-z_TtHYZ54-c6nnKAS1LQnXJ7_6SvIhOMUFIs59DBqSPjExVl0kilRg8WshpGaiDRmc0R7KKWD0myHCT9Yz3IQuzeLF9pXbeVKAxWJcSROBfqyv0pA)

In this part, we created containers for each service, frontend and backend. After that, we publish the container on docker hub, so everyone can pull them for future implementation.  
Moreover, we also provide a docker-compose.yml file to run these 2 together.

  

# RESULTS

### Model

1. #### Resnet50 + unet

![](https://lh7-us.googleusercontent.com/G8Mls5UmMx8l7GBMvFntjNUaVqzNvW6sOsyqung92gXeUlxUoPkEpPcUK_HXZdcT3WH4dKK_a28MjW6Vl0v573Lv5tzVflU1lKi14i7bxS4RZ8ugkuOiP1jJeytKknAwF7iBTB8_LSVmxSJFUXv3AGA)  
For this model, the precision is around 57% which is not good for medical segmentation that needs to be precise to avoid misdiagnosis but for the recall it’s score 82.12% which is quite good to detect all the tumors.

  

2. #### Vgg16 + unet

![](https://lh7-us.googleusercontent.com/bP37PghXODkvbOy3lAgL1hgMWeGor9C6HN6qZb4XhEfGbNbgu64Gz9vBHkDzIMqJcrs6fiRAX4HF1t1uyTptLArjwS91GUQ-Y3wC3gznvq5Xi7_QvrWRaclLQuDbtZoY_2hJpSu1Oo6ynKv4U9kvIIE)

For this model, the precision is around 65.63% which is better than previous model and the recall score 97.07% which is good.
|  |Accuracy  | Dice coefficient | BCE Loss | Precision | Recall |
|--|--|--|--|--|--|
| ResNet50 + Unet | 0.81 |	0.67 |  0.64| 0.57 | 0.82 |
| VGG16 + Unet |0.87  |	0.78 | 0.68 | 0.66 | 0.97 |

In conclusion, Unet with VGG16 as encoder is segmented better than Unet with ResNet50 and both of them have to adjust for misdiagnosis (Precision) because they are likely to predict the areas as tumor more easily than another.

  

### FastAPI

In https://localhost:8000/docs, you can see 2 apis and both of them can be tested by following steps.

1.  On your selected api, click try it out.
    

![](https://lh7-us.googleusercontent.com/6ibzjzWs-86upNl9ABoPtRduwlhbHK2gWgz2yh9VOiybz6WDOz22xuUTwTYcOGt5dHs9HlAWOW9H4Nre9rfYaOqgmZ7QrgeN2b09aGMGsXghsH-Ec8sPUuZUNm97_tFuFmSFMP900EzhZBEXemGLikY)

2.  The browse button will pop up, then select png image that you would like to process the segmentation
    

![](https://lh7-us.googleusercontent.com/9JmM0Ba7odThnlIGhq0dQTbBCchsBl6SAE1iCqoLrTxciJt_W0y3D7NzccMZ1oDZjlZ4wxJYur5qDiSXYtULFeUn6WlWrzTXr6lIykKsBPo2RJS7GxpCCMxSByauqN-AheQUqRGg8zUl3DUcnEis7RA)

  

3.  After waiting for some time, the result will show up in the responses section.
    

![](https://lh7-us.googleusercontent.com/8t5-eG_XyWIPxlwtH-bCkQefW0wt7_V5QtQjalKYnv9TWPf4rjVmCopPbaToy4hdlf0aUIepJ6ch0DuYjZgSXKVESEQfeQny7GJmFrLs-bcigdmfk1WBP-gzOoR0g_QI_GUyqLlTFMzRkHJ9PgYyP8A)

### Frontend

This is the interface of our web that you can upload the image to see segmentation results of our 2 models. To upload the image on our web, please follow this step.

![](https://lh7-us.googleusercontent.com/wvdhwzw5kG93gLWm0__w-gRbX-yYAZD-wyCJQMhz5vhSCE6MPM3cU2OX9F9DQQdl3Es9njYtpSCF-M5HGRZkQpA8-aTei9amJAljubnEz7CfVz6H_eDSpg1qzthZKwLQouF9xRJuG407L5dRTlErPFM)![](https://lh7-us.googleusercontent.com/br-rJHCsM5Hblb6kXYYGjvYdl_FKvT29LVOItipAzO5dHvb9rO1vfv011dbXFG1ZcA8Jvh0NjDpnBkuqVEjvmHH55kE_OvFORdyD9BX2WUAW4R21d4og3yjiPHp4QColC07MDc9SiFJUmiKX2eG00qg)

1.  To upload the image click the browse button then select the .png image.
    

![](https://lh7-us.googleusercontent.com/DlDJX2rvcdWXOm07-2CO3tBcBLEmCtpVa9G3Jg_UmQ7ux_0U5Zcn8uOn2lbqHqgcnfBseNfOzcsM0X_MIo01lPFtPAQtAZowqTfV-NiRjSs4j-dhK1apkiJP2xImgbMgRD-QtJrhByF-dGNrATe8nOc)

2.  After that, click upload image to send the image to processing. The process will take some time to run the model. Then the result of segmentation will show up. Pink area is a tumor that segmented from the select model.
    

![](https://lh7-us.googleusercontent.com/-bYMAL5YoLfmCdW0tI5-aNHTa89XofQsfCGSQDbG7gAufXClYFS74Cpiw_w17bTPO2YEwdlgMnr1CbcZnrTMHuKgdltrvrdY0lyV3heThlacp-Q4GiwIg_CH0EFM-QqWRB-6kXhl0Ac4_neSxPXq5m8)

### Docker

Our result of frontend and backend is published on docker hub. You can pull them by following step

 **FastAPI**

1.  using this following code to pull the image [FastAPI Container](https://hub.docker.com/r/mintyani/hackathon-server)

    

```docker pull mintyani/hackathon-server:latest```

2.  To run the image using this command
    

```docker run -p 8000:8000 mintyani/hackathon-server:latest```

3.  Now, you can access our ml document by go to [https://localhost:80000/docs](https://localhost:80000/docs)
    

From this document, you can try to post the image to see the segment result, which we provided 2 models which is resnet50_unet and vgg16_unet

**Web Interface**

1.  using this following code to pull the image [React Container](https://hub.docker.com/r/parindapannoon/hackathon-frontend)
    

```docker pull parindapannoon/hackathon-frontend:latest```

2.  To run the image using this command
    

```docker run -p 3000:3000 parindapannoon/hackathon-frontend:latest```

3.  Now, you can access our web interface by go to [https://localhost:3000](https://localhost:3000)
    

  

**Full website**

1.  Download [docker-compose.yml](https://github.com/YanikaD/Docker-AI-ML-Hackathon-2023/blob/main/docker-compose.yml) file from our github
    
2.  Run this command:
    ```docker-compose up```

3.  Now, you can access our web by go to [https://localhost:3000](https://localhost:3000)
# **Limitations**
1.  Lack of computer storage, sometimes the docker desktop crashes because the storage is full.
2.  The model takes a lot of time to training. so, the colab always crashed.
3.  The model takes some time to predict. especially, the first time we have pull the image from docker hub.
