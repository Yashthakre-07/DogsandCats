import os, random
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Base directory of this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Flask app
app = Flask(__name__)


UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

import os, gdown

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.h5')
DRIVE_LINK = 'https://drive.google.com/file/d/13jkjHg5uc2f_RoIMd_5m7uBQuZxuHwxz/view?usp=sharing'

if not os.path.exists(MODEL_PATH):
    print("→ Downloading model.h5 from Google Drive …")
    gdown.download(DRIVE_LINK, MODEL_PATH, quiet=False)

from tensorflow.keras.models import load_model
model = load_model(MODEL_PATH)
IMAGE_SIZE = (150, 150)

# Prepare image for prediction
def prepare_image(img_path):
    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    x = image.img_to_array(img) / 255.0
    return np.expand_dims(x, axis=0)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    filename = None

    def get_label_and_prob(probs):
        # Handle both single-sigmoid and two-output models
        if probs.shape[0] == 1:
            # Single sigmoid: output is probability of 'Dog'
            dog_prob = probs[0]
            cat_prob = 1.0 - dog_prob
        else:
            # Softmax two outputs: probs[0]=dog, probs[1]=cat
            dog_prob, cat_prob = probs[0], probs[1]
        if dog_prob > cat_prob:
            return 'Dog', dog_prob
        else:
            return 'Cat', cat_prob

    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            filename = file.filename
            save_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(save_path)

            raw = model.predict(prepare_image(save_path))[0]
            label, prob = get_label_and_prob(raw)
            prediction = f"Prediction: {label} ({prob * 100:.2f}%)"

    elif request.args.get('random'):
        rnd_dir = random.choice(['cats', 'dogs'])
        rnd_folder = os.path.join(BASE_DIR, 'static', 'random', rnd_dir)
        filename = random.choice(os.listdir(rnd_folder))
        save_path = os.path.join(rnd_folder, filename)

        raw = model.predict(prepare_image(save_path))[0]
        label, prob = get_label_and_prob(raw)
        prediction = f"Prediction: {label} ({prob * 100:.2f}%)"

    return render_template('index.html', prediction=prediction, filename=filename)

if __name__ == '__main__':
    os.chdir(BASE_DIR)
    app.run(debug=True)