###s######################################################################################################
# Id            : V1.0 4 - make_api_requests.py          
# Type          : Util
# Tests         : None
# Description   : Module to make API requests
#########################################################################################################

import requests
import pandas as pd

def make_api_request(data):
    # url for api
    url = 'http://127.0.0.1:10001/api'
    # make post request
    r = requests.post( url,pd.DataFrame(data).to_json() )
    print r.text
    return r.text
    
product_decriptions = { 'product_decription': ['32gb','SIEMENS','Clothes','Soft','Transcend',"color","Augen","pen drive","soap","water","Bild","Fahrzeug"] }
make_api_request( product_decriptions )
