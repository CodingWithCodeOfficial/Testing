# ML Model Web Predictor

A simple web app to make predictions using your TensorFlow model.

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install flask flask-cors tensorflow pillow numpy
```

### 2. Update Model Path

Edit `backend.py` line 11:
```python
model = tf.keras.models.load_model('your_model.h5')
```
Change `'your_model.h5'` to the path of your actual model file.

### 3. Update Input Size (if needed)

Edit `backend.py` line 29:
```python
img = img.resize((224, 224))
```
Change `(224, 224)` to match your model's expected input dimensions.

### 4. Run the Backend
```bash
python backend.py
```

You should see:
```
* Running on http://127.0.0.1:5000
```

### 5. Open the Frontend

Simply open `frontend.html` in your web browser (double-click the file).

### 6. Upload and Predict

Click "Upload Image for Prediction" and select an image to test your model!

## Troubleshooting

**CORS Error:** Make sure `flask-cors` is installed and the backend is running.

**Connection Error:** Verify the backend is running on port 5000.

**Model Error:** Check that your model path is correct and the model loads successfully.

**Image Size Error:** Ensure the resize dimensions match your model's expected input shape.
