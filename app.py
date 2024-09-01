from fastapi import FastAPI, File,  UploadFile
import requests
from fastapi.middleware.cors import CORSMiddleware
from flask import Flask, redirect, url_for, request, render_template

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
# import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import InputLayer
from fastapi.responses import StreamingResponse


app = FastAPI()

app.mount("/assets", StaticFiles(directory="assets"), name="assets")
app.mount("/static", StaticFiles(directory="static"), name="static")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory='templates')

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="autoencoder8bit.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

def read_file_as_image(data) -> np.ndarray:
    img = Image.open(BytesIO(data))
    img = img.resize((128, 128))
    img = np.array(img) / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    ip_image = read_file_as_image(await file.read())
    
    # Set the input tensor with the preprocessed image
    interpreter.set_tensor(input_details[0]['index'], ip_image.astype(np.float32))
    
    # Run inference
    interpreter.invoke()
    
    # Get the output tensor
    predictions = interpreter.get_tensor(output_details[0]['index'])
    
    # Convert the predictions back to the image format
    output_array = predictions[0]  # Remove the batch dimension
    output_image = (output_array * 255).astype(np.uint8)  # Scale back to 0-255
    output_image_pil = Image.fromarray(output_image)
    
    buf = BytesIO()
    output_image_pil.save(buf, format='PNG')
    buf.seek(0)
    
    return StreamingResponse(buf, media_type="image/png")



# if __name__ == "__main__":
#     uvicorn.run(app, host='localhost', port=8000)