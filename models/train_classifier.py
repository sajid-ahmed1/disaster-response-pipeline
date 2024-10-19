import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sqlalchemy import create_engine
import joblib

def load_data(database_filepath):
    '''
    returns: variables X and y used for machine learning pipeline
    
    input: 
        database_filepath: the database_filepath from which the dataframe is stored
    
    output: 
        X: explanatory variables
        y: response variable
        category names: category names from the dataframe
    '''
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('df', engine)
    X = df['message']
    y = df.drop(['message','id','genre','original'], axis=1)
    category_names = list(y.columns)
    return X, y, category_names


def tokenize(text):
    '''
    returns: breaks each sentence in text into words with stopwords removed and words lemantised (i.e taken back to their root word - such as running -> run)
    
    output: 
        tokens: strings of words 
    
    input: 
        text: the sentence or corpus of text
    '''
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens

def build_model():
    '''
    returns: model trained with fit, transform variabeles within
    
    output: 
        cv: GridsearchCV object
    
    input: none
    '''
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # Define parameters for grid search
    parameters = {
        'clf__estimator__n_estimators': [50, 100]
        #'clf__estimator__min_samples_split': [2, 4]
    }

    # Create grid search object
    cv = GridSearchCV(pipeline, param_grid=parameters,verbose=2)
    return cv


def evaluate_model(model, X_test, y_test, category_names):
    '''
    returns: evaluation of the model with a classification report
    
    output: 
        print statement: classification report by each of the 36 columns
    
    input: 
        model: the GridSearchCV mode
        X_test: dataset used for training
        y_test: to compare y_predicted against
        category_names: from dataframe
    '''
    # Print best parameters
    print("Best parameters:", model.best_params_)

    # Use best estimator for predictions
    y_pred = model.best_estimator_.predict(X_test)

    # Convert y_pred to DataFrame
    y_pred = pd.DataFrame(y_pred, columns=y_test.columns)

    # Print classification report for each column
    for column in y_test.columns:
        print(f"Classification report for {column}:")
        print(classification_report(y_test[column], y_pred[column]))
        print("\n")


def save_model(model, model_filepath):
    '''
    saves model under model_filepath
    '''
    model = model.best_estimator_
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()