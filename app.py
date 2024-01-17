from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import base64

app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model('banan-v5.h5')

# Define a function to preprocess the image for prediction
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0  # Normalize pixel values to between 0 and 1

# Define a route to serve the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Define a route to handle image classification requests
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the image file from the request
        image_file = request.files['file']

        # Save the image temporarily
        image_path = 'temp_image.jpg'
        image_file.save(image_path)

        # Preprocess the image
        img_array = preprocess_image(image_path)

        # Make predictions
        predictions = model.predict(img_array)

        # Get the predicted class (replace this with your class labels)
        class_labels = ['overripe (สุกเกินไป)', 'ripe (สุก)', 'rotten (สุกงอม)', 'underripe (ยังไม่สุก )']
        predicted_class = class_labels[np.argmax(predictions)]

        # Return the result as JSON
        result = {'class': predicted_class, 'confidence': float(predictions.max()) * 100}
        return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
