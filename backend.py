from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Load your model
model = tf.keras.models.load_model('cifar10_cnn.h5')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img = Image.open(io.BytesIO(file.read()))
    
    # Preprocess (adjust for your model)
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    predictions = model.predict(img_array)
    
    return jsonify({
        'predictions': predictions.tolist(),
        'class': int(np.argmax(predictions)),
        'confidence': float(np.max(predictions))
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
