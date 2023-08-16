from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model, label encoder, and standard scaler
model = joblib.load('model.pkl')
label_encoder = joblib.load('label_encoder.pkl')
scaler = joblib.load('standard_scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the form
    category = request.form['category']
    view_count = float(request.form['view_count'])
    likes = float(request.form['likes'])
    comment_count = float(request.form['comment_count'])
    

    # Encode the category using the label encoder
    category_encoded = label_encoder.transform([category])[0]

    # Scale the input data using the standard scaler
    scaled_data = scaler.transform([[view_count, likes, comment_count]])

    # Combine the data for prediction
    input_data = np.hstack(([[category_encoded]], scaled_data))

    # Make a prediction using the trained model
    prediction = int(model.predict(input_data))

    return render_template('result.html', prediction_text=f'Predicted Dislikes: {prediction}')

if __name__ == '__main__':
    app.run(debug=True)
