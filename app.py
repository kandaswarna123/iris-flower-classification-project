import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from flask import Flask, render_template, request, jsonify

# Data preparation and model training
data = pd.read_csv("Irisdataset.csv")
data.dropna(inplace=True)

le = LabelEncoder()
data["Species"] = le.fit_transform(data["Species"])

X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = data['Species']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

model = LogisticRegression(max_iter=200)
model.fit(x_train, y_train)

# Flask application
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

# @app.route('/enter.html')
# def enter():
#     return render_template('enter.html')

@app.route('/about.html')
def about():
    return render_template('about.html')

@app.route('/login.html')
def login():
    return render_template('login.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        features = [float(x) for x in request.form.values()]
        final_features = [np.array(features)]
        prediction = model.predict(final_features)
        
        output = le.inverse_transform(prediction)[0]
        
        return render_template('login.html', prediction_text=f'The predicted Iris species is {output}')
    
    return render_template('login.html')

if __name__ == '__main__':
    app.run(debug=True)