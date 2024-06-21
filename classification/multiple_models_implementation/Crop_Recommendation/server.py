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

scaler = pickle.load(open('Scaler.pkl', 'rb'))
model = pickle.load(open('crop_recommendation.pkl', 'rb'))

@app.route('/predict', methods = ['POST'])
@cross_origin()
def predict():

    # read the json data
    data = request.get_json(force=True)

    listd = [data['Nitrogen'], data['Potassium'], data['Temperature'], data['Humidity'], data['pH_Value'], data['Rainfall']]

    newdata_df = pd.DataFrame([listd], columns = ['Nitrogen', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall'])

    newdata_df = scaler.transform(newdata_df)

    # convert json data into the numpy array
    #np_data = np.array(listd)
    #print('Numpy Data: ', np_data)

    prediction = model.predict(newdata_df).tolist()
    print(prediction)
    
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(port=5000,debug=True)
    