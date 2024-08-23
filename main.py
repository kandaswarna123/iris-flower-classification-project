import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from flask import Flask, render_template, request


# Data preparation and model training
data = pd.read_csv("C:\\Users\\swarn\\OneDrive\\Desktop\\pro1\\iris flower classification\\Irisdataset.csv")
print(data.head())
print("Missing values:")
print(data.isnull().sum())

data.dropna(inplace=True)

le = LabelEncoder()
data["Species"] = le.fit_transform(data["Species"])

X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = data['Species']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

model = LogisticRegression(max_iter=200)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

f = [[2,3,3,2]]
prediction = model.predict(f)[0]


print(f"Predicted Iris class: {prediction}")
