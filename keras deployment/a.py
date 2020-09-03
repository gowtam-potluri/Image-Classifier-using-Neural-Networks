import os
import sys

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Some utilites
import numpy as np
#from util import base64_to_pil
import re
from PIL import Image
from io import BytesIO
import base64
import re
import cv2
import numpy as np


# Declare a flask app
app = Flask(__name__)


# You can use pretrained model from Keras
# Check https://keras.io/applications/
from keras.applications.mobilenet_v2 import MobileNetV2
model = load_model('model/mod.h5')

print('Model loaded. Check http://127.0.0.1:5000/')


# Model saved with Keras model.save()
MODEL_PATH = 'models/mod.h5'

# Load your own trained model
# model = load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')


def model_predict(img):
    img = img.resize((150, 150))
    img=np.reshape(img,[1,150,150,3])
    '''
    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    
'''
    #img = np.expand_dims(img, axis=0)
    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
   # x = preprocess_input(x, mode='tf')

    preds = model.predict(img)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        #img = base64_to_pil(request.json)
        image_data = re.sub('^data:image/.+;base64,', '', request.json)
        print(image_data)
        img = Image.open(BytesIO(base64.b64decode(image_data)))

        # Save the image to ./uploads
        # img.save("./uploads/image.png")

        # Make prediction
        preds = model_predict(img)
        print(preds)
        if(preds[0]==0.):
            s='Bad Banana'
        else:
            s='Good Banana'
        # Serialize the result, you can add additional fields
        return jsonify(result=s)

    return None


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()