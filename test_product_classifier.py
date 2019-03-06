###s######################################################################################################
# Id            : 4 ML - test_product_classifier.py          
# Type          : Util
# Tests         : 
# Description   : Util to preprocess product classifier data and create machine learning model
#########################################################################################################


from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
import seaborn as sns

import unittest

from product_classifier import PARENT_DIR, MODEL_NAME, TESTSET_FILE, CV, \
    process_data, merge_columns, read_data, persist_model

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

if __name__ == '__main__':    
    unittest.main()