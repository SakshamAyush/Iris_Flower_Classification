# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 16:03:57 2018

@author: Saksham
"""

import flask
from flask import Flask
from sklearn.externals import joblib
from flask import render_template, request
import numpy as np

app = Flask(__name__)

@app.route("/")
@app.route("/index")
def index():
   return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method=='POST':
        f1 = request.form['f1']
        f2 = request.form['f2']
        f3 = request.form['f3']
        f4 = request.form['f4']
        feature_array = [f1,f2,f3,f4]
        feature = np.asarray(feature_array).reshape(1,4)
        #feature_array = request.get_json()['feature_array[]']
        prediction = model.predict(feature).tolist()
        return flask.jsonify(prediction)
if __name__ == '__main__':
    model = joblib.load('iris.pkl')
    app.run(host='0.0.0.0', port=8000, debug=True)
    
