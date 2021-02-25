#!/usr/bin/env python
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from flask import Flask, render_template, request, jsonify
from flask_dropzone import Dropzone
import os

from keras.applications.vgg16 import VGG16
model = VGG16()


def predict(path):
    image = load_img(path, target_size=(224, 224))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # predict the probability across all output classes
    yhat = model.predict(image)
    # convert the probabilities to class labels
    result = decode_predictions(yhat)
    labels = []
    value = []
    for i in range(5):
        labels.append(result[0][i][1])
        value.append(float(result[0][i][2]))
    return labels, value


app = Flask(__name__)
app.config['SECRET_KEY'] = 'my_secret!.'
basedir = os.path.abspath(os.path.dirname(__file__))
os.path.join(os.path.dirname(__file__))

app.config.update(
    UPLOADED_PATH=os.path.join(basedir, 'static//uploads'),
    # Flask-Dropzone config:
    DROPZONE_ALLOWED_FILE_TYPE='image',
    DROPZONE_MAX_FILE_SIZE=3,
    DROPZONE_MAX_FILES=30,
)

dropzone = Dropzone(app)


@app.route('/', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        f = request.files.get('file')
        file_path = os.path.join(app.config['UPLOADED_PATH'], f.filename)
        f.save(file_path)

        labels, value = predict(file_path)
        return jsonify(labels=labels,
                       value=value)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
