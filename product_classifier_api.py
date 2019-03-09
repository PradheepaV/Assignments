###s######################################################################################################
# Id            : V1.0 4 - product_classifier_api.py          
# Type          : Util
# Tests         : None
# Description   : Util to host REST API for product_classifier
#########################################################################################################
 
from flask import Flask, request
import pandas as pd
import numpy as np
import json
import pickle
import os

from product_classifier import PARENT_DIR, MODEL_NAME

app = Flask(__name__)

# Load Model Files
model_filepath = os.path.join( PARENT_DIR, MODEL_NAME)
model = pickle.load( open( model_filepath ) )

@app.route( '/api', methods=[ 'POST' ])
def make_prediction():
    ''' Make prediction based on the input text'''
    # read json object and conver to json string
    data = json.dumps( request.get_json(force=True) )
    # create pandas dataframe using json string
    df = pd.read_json( data )
    product_decriptions = df[ 'product_decription' ].ravel()      
    X = df[ 'product_decription' ]
    # make predictions
    predictions = model.predict( X )
    # create response dataframe
    df_response = pd.DataFrame( {'product_decription': product_decriptions, 'Predicted' : predictions } )
    return df_response.to_json()
 
if __name__ == '__main__':
    # host flask app at port 10001
    app.run(port=10001, debug=True)
