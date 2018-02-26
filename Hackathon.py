#!flask/bin/python

import sys

from flask import Flask, render_template, request, jsonify
import random, json, pickle

import xgboost as xgb
import formatter as fw
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model/prediction_model.pkl','rb'))

def format_data(json_data):
    print('Formatting Test Data to the form of Treatable format for Model')
    test_df, xdmatrix = fw.format_test_data(json_data)
    return test_df, xdmatrix

def predict(test_df, xdmatrix):
    ytest = model.predict(xdmatrix)
    test_df['price'] = np.exp(ytest) - 1
    temp_dict = test_df.to_dict('list')
    return temp_dict['price'][0]

@app.route('/')
def output():
    # serve index template
    return render_template('index.html')
	
@app.route('/receive', methods = ['POST'])
def worker():
    # read json + reply
    data = request.get_json()
    conditions = { "Brand New" : 1, "Refurbished" : 2, "New & Heavily Used" : 3, "Old & Less Used" : 4 , "Old & Heavily Used" :5}

    data["item_condition_id"] = str(conditions[data["item_condition_id"]])
    data["shipping"] = str(1 if data["shipping"] == "Yes" else 0)

    print(data)
    test_df, xdmatrix = format_data(data)
    output = predict(test_df, xdmatrix)
    output = round(output,2)
    print(output)

    return str(output)

if __name__ == '__main__':
    # run!
    app.jinja_env.auto_reload = True
    app.run(port=8080)
