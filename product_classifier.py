###s######################################################################################################
# Id            : 4 ML - product_classifier.py          
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

PARENT_DIR      = "."
MODEL_NAME      = "clf_model.pkl"
TESTSET_FILE    = "testset_C.csv"
CV              = 5

def process_data( df ):
    ''' Pre Process data, balanced dataset'''
    return (df
            .drop( [ 'id' ], axis = 1 )
            .fillna( '' )
            .replace( '\d+', '' ) 
            .assign( product_decription = lambda x: x.main_text + x.add_text + x.manufacturer )
            .drop( [ 'main_text', 'add_text', 'manufacturer' ], axis = 1 )            
    )

def merge_columns( df ):
    ''' Merges columns and returns dataframe'''
    df[ 'product_decription' ] = df[ 'main_text' ] + df[ 'add_text' ] + df[ 'manufacturer' ]
    return df

def read_data( filename, sep=";" ):
    ''' Reads file content and returns dataframe'''
    return pd.read_csv( filename, sep=sep )

def analyze_model( df ):
    ''' Build various model and produce their performance statistics for analysis'''
    X       = df[ 'product_decription' ].ravel()
    y       = df[ 'productgroup' ].ravel()
    
    models = [ ( 'RandomForest', RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0) ) ,
               ( 'LinearSVC',    LinearSVC() ),
               ( 'NB',           MultinomialNB() ),
               ( 'LogReg',       LogisticRegression( n_jobs=1, C=1e5 ) ),
               ( 'SGD',          SGDClassifier( loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None) ),                              
    ]
        
    entries = []
    for each_model in models:
        model_name  = each_model[1].__class__.__name__
        model       = Pipeline([('vect', CountVectorizer()), 
                                ('tfidf', TfidfTransformer()), 
                                ('clf',  each_model[1]), 
                                ]) 
        accuracies  = cross_val_score(model, X, y, scoring='accuracy', cv=CV)
        for fold_idx, accuracy in enumerate(accuracies):
            entries.append((model_name, fold_idx, accuracy))

    cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
    
    sns.boxplot( x='model_name', y='accuracy', data=cv_df )
    sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
                size=8, jitter=True, edgecolor="gray", linewidth=2)
    plt.show()
    print cv_df
    print cv_df.groupby('model_name').accuracy.mean()
    
def build_model( df ):
    ''' Build the selected Model - LinearSVC'''
    lsvc        =  Pipeline([('vect',  CountVectorizer()), 
                    ('tfidf',   TfidfTransformer()), 
                    ('clf',     LinearSVC()),
                   ])
    X           = df[ 'product_decription' ].ravel()
    y           = df[ 'productgroup' ].ravel()
    clf_model   = GridSearchCV( lsvc, param_grid={}, cv=CV )    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    clf_model.fit( X_train, y_train ) 
    y_pred = clf_model.predict( X_test )  
    print('accuracy %s' % accuracy_score(y_pred, y_test)) 
    print(classification_report(y_test, y_pred)) 
    return clf_model

def persist_model( model, modelName="" ):
    ''' Pickle and dump the model locally '''
    model_file_path     = os.path.join( PARENT_DIR, modelName or MODEL_NAME )
    model_file_pickle   = open(model_file_path, 'wb')
    pickle.dump( model, model_file_pickle )
    model_file_pickle.close()    

if __name__ == '__main__':    
    df = read_data( '%s\\%s' %( PARENT_DIR, TESTSET_FILE ) )
    df = process_data( df )
    analyze_model( df )  
    model = build_model( df )  
    persist_model( model )
     
