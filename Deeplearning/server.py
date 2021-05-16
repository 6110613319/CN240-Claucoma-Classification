from fastapi import FastAPI, UploadFile, Form, File
import tensorflow as tf
import cv2
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

def muteluh_model(img):
    # x = image.img_to_array(img)
    x = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    model = load_model('resnet50_best.h5')
    prediction = model(img)
    class_names = ["glaucoma", "normal", "other"]
    argmax = np.argmax(prediction > 0.5).astype("int32")
    score = tf.nn.softmax(prediction[0])
    return class_names[argmax], score[argmax]
    
app = FastAPI()

@app.get("/")
async def helloworld():
    return {"greeting": "Hello World"}


@app.post("/api/fundus")
async def upload_image(nonce: str=Form(None, title="Query Text"), 
                       image: UploadFile = File(...)):
    contents = await image.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img,(256,256))
    img = np.reshape(img,[1,256,256,3])
    
    class_out, class_conf = muteluh_model(img)
    
    return {
        "nonce": nonce,
        "classification": class_out,
        "confidence_score": np.float(class_conf),
        "debug": {
            "image_size": dict(zip(["height", "width", "channels"], img.shape)),
        }
    }
