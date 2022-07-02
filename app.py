#Bicycle Thefts
# COMP247 (sec.003) 
# Group 03 
# Sayed Fayaz (3011099990) 
# Yoo Sun Song 301091906


import sys
from flask import Flask, request, jsonify,redirect, url_for,render_template
import joblib
import traceback
import numpy as np
import json
import numpy as np
import pandas as pd
import os,inspect




basepath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) 

#1. Load the model
Decision_Tree  = joblib.load(basepath + "/group3_dTree_model.pkl")
Logistic_Regression = joblib.load(basepath + "/group3_logistic_model.pkl")
Random_Forest = joblib.load(basepath + "/group3_randomFor_model.pkl")
Neural_Networks = joblib.load(basepath + "/group3_neural_model.pkl")
Support_Victor_Machines  = joblib.load(basepath + "/group3_svm_model.pkl")

## API definition
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/testing")
def testing():
    return render_template("testing.html")

@app.route("/result", methods=['POST'])
def test():
    if request.method == 'POST':
        input_dict = request.form.to_dict()
        model = input_dict.pop('model')
        df = pd.DataFrame.from_dict([input_dict])
    
        if (model == 'Decision_Tree'):
            current_model = Decision_Tree
        elif (model == 'Random_Forest'):
            current_model= Random_Forest
        elif (model == 'Logistic_Regression '):
            current_model = Logistic_Regression
        elif (model == 'Neural_Networks'):  
            current_model = Neural_Networks
        else:
            current_model = Support_Victor_Machines
            
        predict = current_model.predict(df)
    
        print (predict)
    if (predict==0):
        print ('Lost Bicycle')
        return render_template("result.html",model_name= model,prediction=predict, result = "Lost Bicycle")
    else:
        print ('Found Bicycle')
        return render_template("result.html",model_name= model,prediction=predict,result = "Found Bicycle")



if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) 
    except:
        port = 12346  



    app.run(port=port, debug=True)