import pickle
from flask import Flask, request, jsonify, render_template, url_for
import numpy as np

app = Flask(__name__, static_url_path='/static')

# Load the pre-trained model
with open('model_lr.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    # Render the home.html template
    return render_template('home.html')

@app.route('/predict_api', methods=["POST"])
def predict_api():
    # Get the JSON data from the request
    data = request.json['data']
    print("Received data:", data)
    
    # Convert the data into a numpy array and reshape it for the model
    new_data = np.array(list(data.values())).reshape(1, -1)
    print("Data for prediction:", new_data)
    
    # Make a prediction using the model
    output = model.predict(new_data)
    print("Prediction output:", output[0])
    
    # Convert prediction to a regular Python type (e.g., float)
    prediction = output[0].item() if isinstance(output[0], np.float32) else output[0]
    
    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction})

@app.route('/predict', methods=["POST"])
def predict():
    # Extract form data
    form_data = request.form
    
    # Extract first three elements as integers
    int_values = [int(form_data[f"int_{i}"]) for i in range(1, 4)]
    
    # Extract remaining elements as floats
    float_values = [float(form_data[f"float_{i}"]) for i in range(4, len(form_data) + 1)]
    
    # Combine integer and float values
    data = int_values + float_values
    print("Received data:", data)
    
    # Reshape data for prediction
    new_data = np.array(data).reshape(1, -1)
    
    # Make prediction
    output = model.predict(new_data)[0]
    print(f"The predicted value is: {output}")

    # Render home.html with prediction result
    return render_template("home.html", prediction_text=f"The Predicted Price is {output}")

if __name__ == "__main__":
    app.run(debug=True)