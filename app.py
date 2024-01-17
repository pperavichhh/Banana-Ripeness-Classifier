from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import base64

app = Flask(__name__)

model = tf.keras.models.load_model('banan-v5.h5')

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        image_file = request.files['file']
        image_path = 'temp_image.jpg'
        image_file.save(image_path)
        img_array = preprocess_image(image_path)
        
        predictions = model.predict(img_array)
        class_labels = ['overripe (สุกเกินไป)', 'ripe (สุก)', 'rotten (สุกงอม)', 'underripe (ยังไม่สุก )']
        predicted_class = class_labels[np.argmax(predictions)]

        result = {'class': predicted_class, 'confidence': float(predictions.max()) * 100}
        return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
