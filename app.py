
from flask import Flask, request, render_template
import numpy as np
import pickle
import os

# Ensure paths are correct using os.path.join()
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
sc_path = os.path.join(os.path.dirname(__file__), 'standscaler.pkl')
mx_path = os.path.join(os.path.dirname(__file__), 'minmaxscaler.pkl')

# Load model and scalers
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(sc_path, 'rb') as f:
        sc = pickle.load(f)
    with open(mx_path, 'rb') as f:
        mx = pickle.load(f)
except FileNotFoundError as e:
    print("Error:", e)
    exit()

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    try:
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosporus'])
        K = float(request.form['Potassium'])
        temp = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        pH = float(request.form['pH'])
        rainfall = float(request.form['Rainfall'])

        # Input validation
        if N < 0 or P < 0 or K < 0 or temp < -50 or temp > 60 or humidity < 0 or humidity > 100 or pH < 0 or pH > 14 or rainfall < 0:
            result = "Invalid input values. Please ensure all inputs are within realistic ranges."
        else:
            feature_list = [N, P, K, temp, humidity, pH, rainfall]
            single_pred = np.array(feature_list).reshape(1, -1)

            mx_features = mx.transform(single_pred)
            sc_mx_features = sc.transform(mx_features)
            prediction = model.predict(sc_mx_features)

            crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                         8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                         14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                         19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

            if prediction[0] in crop_dict:
                crop = crop_dict[prediction[0]]
                result = "{} is the best crop to be cultivated right there.".format(crop)
            else:
                result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
    except ValueError:
        result = "Invalid input values. Please ensure all inputs are numeric."
    
    return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)










