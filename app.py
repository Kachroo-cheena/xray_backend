from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os
from flask_cors import CORS  # Import Flask-CORS

app = Flask(__name__)
CORS(app) 

# Load model and labels
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

np.set_printoptions(suppress=True)

# Load the model
BASE_DIR = os.path.dirname(os.getcwd())
MODEL_DIR = os.path.join(BASE_DIR, 'model')

# loaded_model = load_model(os.path.join(MODEL_DIR, 'keras_model.h5'), compile=False)
# class_names = open(os.path.join(MODEL_DIR, 'labels.txt'), "r").readlines()

xray_model = load_model("model/cnn_xray_classification_13_10.h5",compile=False)
loaded_model = load_model("model/keras_model.h5", compile=False)
class_names = open("model/labels.txt", 'r').readlines()



def is_xray(image):
    # Resize image to the size expected by the X-ray model
    size = (128, 128)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    image_array = image_array / 255.0  # Normalize as done during training
    image_array = np.expand_dims(image_array, axis=0)  # Create batch of one

    # Predict using the X-ray detection model
    xray_prediction = xray_model.predict(image_array)
    return xray_prediction[0] > 0.5 

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    image = Image.open(file).convert("RGB")
    if is_xray(image):
    # Resize image to 224x224 and normalize it
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        # Create a batch of one image
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        # Predict the class of the image
        prediction = loaded_model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        return jsonify({
            'class_name': class_name.strip(),
            'confidence_score': float(confidence_score)
        })
    return jsonify({
            'class_name': 'this image is not an xray',
            'confidence_score': 1
        })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
