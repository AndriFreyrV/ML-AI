from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import color
from skimage import io
import requests
import os
from flask import Flask, render_template, redirect, request, url_for
from werkzeug.utils import secure_filename
import base64
import io


app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'static/'

# load model from training
model = keras.models.load_model('model_gray_1')

locations = ['Douglas Reef', 'Iniskin Bay', 'Kayak Island', 'Kukak Bay', 'Uganik Bay', 'Landlocked Bay',
             'Stockdale Harbor', 'Rocky Bay']

# map of geographic locations of regions
geo_map = {
    'Iniskin Bay': (59.7149, -153.412643989883),
    'Kayak Island': (59.89913355, -144.45194885740182),
    'Kukak Bay': (58.3072222, -154.2488889),
    'Uganik Bay': (57.834848449999996, -153.53225888017403),
    'Landlocked Bay': (60.8344444, -146.5872222),
    'Rocky Bay': (56.055, -132.5822222),
    'Douglas Reef': (58.762233, -153.2725788),
    'Stockdale Harbor': (60.3144485, -147.1909393)
    }

resize_and_rescale = keras.Sequential([
  keras.layers.experimental.preprocessing.Resizing(280, 400),
  keras.layers.experimental.preprocessing.Rescaling(1./255)
])

def test_process(img):
    img_out = color.rgb2gray(img)
    img_out *= 255
    img_out.astype(np.uint8)
    img_resize = resize_and_rescale(color.gray2rgb(img_out))
    return img_resize

def predict(img):
    img_arr = np.array(img)
    features = test_process(img_arr)
    features = np.expand_dims(features, 0)
    return locations[np.argmax(model.predict(features)[0])]


@app.route('/')
def index():
    return render_template('index.html', upload = True, img = False, pred = False)

@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        print('no File')
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    img = Image.open(file)
    pred = predict(img) # predict spawning grounds for image
    data = io.BytesIO()
    img.save(data, "jpeg")
    encoded_img_data = base64.b64encode(data.getvalue())

    loc = list(geo_map[pred]) # location
    return render_template('index.html', upload = False, img = encoded_img_data.decode('utf-8'), pred = pred, loc = loc)


if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8080)