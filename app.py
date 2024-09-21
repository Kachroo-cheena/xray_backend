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

loaded_model = load_model("c:/Users/lenovo/OneDrive/Desktop/insightx_v1/X-ray-Image-Classification/xray_tryout/x-ray-body-part-classification/model/keras_model.h5", compile=False)
class_names = open("c:/Users/lenovo/OneDrive/Desktop/insightx_v1/X-ray-Image-Classification/xray_tryout/x-ray-body-part-classification/model/labels.txt", 'r').readlines()

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    image = Image.open(file).convert("RGB")

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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
