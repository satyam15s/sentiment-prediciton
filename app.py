import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence

app = Flask(__name__)

# Load the trained model
model = load_model("Emotion_Voice_Detection_Model.h5")
model._make_predict_function()  # Necessary for TensorFlow models when running with multiple threads

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Handle the voice input and predict stress
    voice_input = request.files['voice_input']
    
    # Save the voice input as a temporary file
    tmp_file = 'tmp.wav'
    voice_input.save(tmp_file)
    
    # Preprocess the voice input
    # You'll need to use a speech recognition library to convert audio to text if necessary
    
    # Placeholder code for feature extraction and normalization
    # Replace this with the actual preprocessing code specific to your model
    # For example, you might need to extract audio features, convert them to spectrograms, etc.
    preprocessed_input = preprocess_audio(tmp_file)
    
    # Perform prediction
    # You'll need to adapt this code depending on the input requirements of your model
    # The example assumes the model expects a 2D input of shape (time_steps, num_features)
    preprocessed_input = sequence.pad_sequences([preprocessed_input], maxlen=100)  # Adjust maxlen as needed
    prediction = model.predict(preprocessed_input)
    
    # Convert the prediction to a stress level (e.g., Low, Medium, High)
    stress_prediction = convert_prediction_to_stress_level(prediction)
    
    # Delete the temporary file
    os.remove(tmp_file)
    
    return render_template('result.html', stress_prediction=stress_prediction)

if __name__ == '__main__':
    app.run(debug=True)
