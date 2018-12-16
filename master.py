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

## Reading data

d = pd.read_csv('data/train.csv')
Y = d['target']
X = d['question_text']

### Splitting to train and test sets

X_train, X_test, Y_train, Y_test = train_test_split(X, 
                                                    Y, 
                                                    test_size = 0.3, 
                                                    random_state = 1)

## Preprocessing

