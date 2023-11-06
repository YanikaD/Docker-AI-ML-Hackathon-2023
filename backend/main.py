from model import resnet50unet_model,vgg16unet_model
from typing import Union
import numpy as np
from fastapi import FastAPI, UploadFile
import io
from PIL import Image
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS
origins = ["http://localhost:*", "http://localhost:8000", ]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/resnet50_unet")
def Resnet50Unet(image: UploadFile):
    content = image.file.read()
    image = Image.open(io.BytesIO(content))

    result = resnet50unet_model(image,'./resnet50_unet.pt')
    # Convert the PyTorch Tensor to a NumPy array
    result_numpy = result.detach().numpy()

    # Assuming the result is grayscale, duplicate the channel to create an RGB image
    result_rgb = np.stack(result_numpy , axis=-1)

    # Convert the NumPy array to a PIL Image
    result_pil = Image.fromarray(np.uint8(result_rgb))

    # Convert the result image to bytes
    image_bytes = io.BytesIO()
    result_pil.save(image_bytes, format='PNG')
    image_bytes.seek(0)

    # Return the image as a FastAPI StreamingResponse
    return StreamingResponse(image_bytes, media_type="image/png")

@app.post("/vgg16_unet")
def VGG16Unet(image: UploadFile):
    content = image.file.read()
    image = Image.open(io.BytesIO(content))

    result = vgg16unet_model(image,'./vgg16_unet.pt')
    # Convert the PyTorch Tensor to a NumPy array
    result_numpy = result.detach().numpy()

    # Assuming the result is grayscale, duplicate the channel to create an RGB image
    result_rgb = np.stack(result_numpy, axis=-1)

    # Convert the NumPy array to a PIL Image
    result_pil = Image.fromarray(np.uint8(result_rgb))

    # Convert the result image to bytes
    image_bytes = io.BytesIO()
    result_pil.save(image_bytes, format='PNG')
    image_bytes.seek(0)

    # Return the image as a FastAPI StreamingResponse
    return StreamingResponse(image_bytes, media_type="image/png")