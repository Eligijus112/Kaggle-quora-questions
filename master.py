"""
Pipeline: 
    read data ->
    preprocess ->
    create model ->
    forecast    
"""

## Importing modules and functions

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import xgboost as xgb

from preprocess import clean

## Hyper parameters

test_share = 0.3

## Reading data

d = pd.read_csv('data/train.csv')
Y = d['target']
X = d['question_text']

### Splitting to train and test sets

if(test_share is not 0):
    X_train, X_test, Y_train, Y_test = train_test_split(X, 
                                                        Y, 
                                                        test_size = test_share, 
                                                        random_state = 1)
if(test_share == 0):
    X_train = X
    Y_train = Y    
    d_test = pd.read_csv('data/test.csv')
    X_test = d_test['question_text']    
    
## Preprocessing pipeline

def preproc_pipeline(string_df):
    string_df = clean.to_lower(string_df)
    string_df = clean.rm_punctuations(string_df)
    string_df = clean.rm_digits(string_df)
    string_df = clean.rm_stop_words(string_df)
    string_df = clean.clean_ws(string_df)
    return string_df

X_train = preproc_pipeline(X_train)

### Creating a document term matrix

vect = CountVectorizer(min_df = 0.00005)
dtm_train = vect.fit_transform(X_train)

## Creating a model

### boosting 

xg_log = xgb.XGBClassifier(objective ='reg:logistic', 
                           colsample_bytree = 0.3, 
                          learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 20)
fitted_xgboost = xg_log.fit(dtm_train, Y_train)

### Logistic

model = LogisticRegression(solver = 'newton-cg')
fitted_model = model.fit(dtm_train, Y_train)

### Creating a dataframe to store the coefficients and features

ft_names = vect.get_feature_names()
coefficients = fitted_model.coef_[0]
coef_df = pd.DataFrame({'feature' : ft_names, 
                        'coefficient' : coefficients.tolist()})
coef_df = coef_df.sort_values('coefficient')    

## Forecasting on test data 

### Preprocesing test data 

X_test = preproc_pipeline(X_test)

### Creating the test document term matrix

vect = CountVectorizer(vocabulary = ft_names)
dtm_test = vect.fit_transform(X_test)

### Prediction

y_hat = fitted_model.predict(dtm_test)
y_hat_boost = fitted_xgboost.predict(dtm_test)

if(test_share is not 0):
    
    ### Calculating accuracy
    
    y_actual = Y_test.tolist()
    y_hat = y_hat.tolist()
    print('Logistic regression:')
    print(f1_score(y_actual, y_hat))
    
    y_hat_boost = y_hat_boost.tolist()
    print('Xgboost:')
    print(f1_score(y_actual, y_hat_boost))

if(test_share == 0):
    
    ## Saving the final predictions
    d_test = d_test.reset_index()
    d_test['prediction'] = y_hat
    to_upload = d_test[['qid', 'prediction']]
    to_upload.to_csv('output/submission.csv', index = False)