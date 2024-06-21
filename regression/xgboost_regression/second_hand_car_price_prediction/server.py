import os
import numpy as np
import flask
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

model = pickle.load(open('second_hand_car_price_model.pkl', 'rb'))

@app.route('/predict', methods = ['POST'])
@cross_origin()
def predict():

    # read the json data
    data = request.get_json(force=True)

    listd = [data['Brand'], data['Model'], data['Kilometers_Driven'], data['Fuel_Type'], data['Transmission'], data['Owner_Type'], data['Mileage'], data['Engine'], data['Power'], data['Car_Age']]

    # convert json data into the numpy array
    np_data = np.array(listd)
    print('Numpy Data: ', np_data)

    prediction = model.predict([np_data]).tolist()
    print(prediction)
    
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(port=5000,debug=True)
    