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

from preprocess import clean

## Reading data

d = pd.read_csv('data/train.csv')
Y = d['target']
X = d['question_text']

### Splitting to train and test sets

X_train, X_test, Y_train, Y_test = train_test_split(X, 
                                                    Y, 
                                                    test_size = 0.3, 
                                                    random_state = 1)

## Preprocessing pipeline

def preproc_pipeline(string_df):
    string_df = clean.to_lower(X_train)
    string_df = clean.rm_punctuations(string_df)
    string_df = clean.rm_digits(string_df)
    string_df = clean.rm_stop_words(string_df)
    string_df = clean.clean_ws(string_df)
    return string_df

X_train = preproc_pipeline(X_train)

### Creating a document term matrix

vect = CountVectorizer(min_df = 0.0001, 
                       binary = True, 
                       ngram_range = (1, 2))
dtm_train = vect.fit_transform(X_train)

## Creating a model

model = LogisticRegression()
fitted_model = model.fit(dtm_train, Y_train)

### Creating a dataframe to store the coefficients and features

ft_names = vect.get_feature_names()
coefficients = fitted_model.coef_[0]
coef_df = pd.DataFrame({'feature' : ft_names, 
                        'coefficient' : coefficients.tolist()})
coef_df = coef_df.sort_values('coefficient')    

## Forecasting on test data 

### Creating the test document term matrix

vect = CountVectorizer(vocabulary = ft_names, 
                       binary = True, 
                       ngram_range = (1, 2))
dtm_test = vect.fit_transform(X_test)

### Prediction

y_hat = fitted_model.predict(dtm_test)

### Calculating accuracy

y_actual = Y_test.tolist()
y_hat = y_hat.tolist()
print(f1_score(y_actual, y_hat))
