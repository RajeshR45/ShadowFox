from flask import Flask, request, jsonify
from flask_cors import CORS  
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from io import BytesIO

app = Flask(__name__)
CORS(app) 

model_path = 'D://shadow fox//model.keras'

try:
    model = load_model(model_path)  
    print("Model loaded successfully.")
except Exception as e:
    print("Error occurred while loading model",str(e))
    model = None 

class_labels = ['car', 'cat']  
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model could not be loaded. Please check the model file path.'}), 500

    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image'] 

        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400


        img_bytes = BytesIO(file.read())
        image = load_img(img_bytes, target_size=(128, 128))  
        image_array = img_to_array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)  
        predictions = model.predict(image_array)
        predicted_class = class_labels[np.argmax(predictions)] 
        confidence = np.max(predictions)  
        print(f"Predicted Class: {predicted_class}, Confidence: {confidence}")

        return jsonify({'predicted_class': predicted_class, 'confidence': f'{confidence:.2f}'})  

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({'error': 'An error occurred during prediction', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
