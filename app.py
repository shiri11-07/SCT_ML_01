from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open('house_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sqft = float(request.form['sqft'])
    beds = int(request.form['bedrooms'])
    baths = int(request.form['bathrooms'])

    features = np.array([[sqft, beds, baths]])
    prediction = model.predict(features)[0]

    return render_template('index.html', prediction_text=f'Estimated House Price: ${prediction:,.2f}')

if __name__ == '__main__':
    app.run(debug=True)
