###s######################################################################################################
# Id            : V1.0 4 ML - test_product_classifier.py          
# Type          : Util
# Tests         : 
# Description   : Util to test product_classifier
#########################################################################################################

import pandas as pd
import pickle
import os
import unittest
from product_classifier import PARENT_DIR, MODEL_NAME, TESTSET_FILE, CV, \
    process_data, merge_columns, read_data, persist_model, analyze_model, build_model
from mock import patch
import matplotlib.pyplot

class ProductClassifierTests(unittest.TestCase): 

    def setUp(self): 
        self.df = pd.read_csv( '%s\\%s' %( PARENT_DIR, TESTSET_FILE ) , sep=";" ) 
  
    def test_process_data(self):         
        self.assertTrue( process_data(self.df) is not None )
    
    def test_merge_columns(self):         
        self.assertTrue( merge_columns(self.df)['product_decription'] is not None ) 

    def test_read_data( self ):    
        self.assertTrue( read_data( '%s\\%s' %( PARENT_DIR, TESTSET_FILE ) ) is not None ) 
    
    def test_persist_model( self ):
        persist_model( self.df, modelName="Dummy" )#filename to be datetime stamped to be unique everytime
        self.assertTrue( os.path.isfile(os.path.join( PARENT_DIR, "Dummy" )) )
    
    @patch('matplotlib.pyplot.show')
    def test_analyze_model( self, patch_plt ):
        ''' Ensure it runs without any error'''
        analyze_model( process_data(self.df) )    
    
    def test_build_model( self ):
        '''Ensure it runs without any error'''
        build_model( process_data(self.df) )

        
if __name__ == '__main__':    
    unittest.main()
