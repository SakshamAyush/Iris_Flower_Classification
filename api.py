# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 16:03:57 2018

@author: Saksham
"""

import flask
from flask import Flask
from sklearn.externals import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
@app.route("/index")
def index():
   return flask.render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
    
if __name__ == '__main__':
    model = joblib.load('model.pkl')
    app.run(host='0.0.0.0', port=8000, debug=True)
    
@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method=='POST':
        return render_template('index.html', label="3")    