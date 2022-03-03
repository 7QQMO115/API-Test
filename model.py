"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso


def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.
    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])
    data = pd.read_csv('./utils/data/df_test.csv', index_col=0)
    feature_vector_df = data

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
        
    feature_vector_df['time'] = pd.to_datetime(feature_vector_df['time'])
    feature_vector_df['year'] = pd.to_datetime(feature_vector_df['time']).dt.year
    feature_vector_df['month_of_year'] = pd.to_datetime(feature_vector_df['time']).dt.month
    feature_vector_df['week_of_year'] = feature_vector_df['time'].dt.isocalendar().week.astype(int)
    feature_vector_df['day_of_month'] = pd.DatetimeIndex(feature_vector_df['time']).day
    feature_vector_df['day_of_week'] = pd.DatetimeIndex(feature_vector_df['time']).dayofweek
    feature_vector_df['day_of_year'] = pd.DatetimeIndex(feature_vector_df['time']).dayofyear
    feature_vector_df['hour_of_day'] = pd.DatetimeIndex(feature_vector_df['time']).hour
    feature_vector_df['hour_of_the_week'] = feature_vector_df['day_of_week'] * 24 + (feature_vector_df['hour_of_day'] + 1)
    feature_vector_df['hour_of_year'] = feature_vector_df['day_of_year'] * 24 + feature_vector_df['hour_of_day']
    
    feature_vector_df = pd.get_dummies(feature_vector_df, drop_first=True)
    #feature_vector_df = feature_vector_df.fillna(feature_vector_df['Valencia_pressure'].mean())
    
    test_names = list(feature_vector_df.columns)
    test_names.remove('time')
    test_data = feature_vector_df[test_names]
    
    #y_train = feature_vector_df['load_shortfall_3h']
    #X_train = feature_vector_df.drop('load_shortfall_3h', axis=1)
    #X_train = pd.get_dummies(X_train, columns=['Valencia_wind_deg', 'Seville_pressure'], drop_first=True)

    scaler = StandardScaler()
    # create scaled version of the predictors (there is no need to scale the response)
    scaled = scaler.fit_transform(test_data)
    # convert the scaled predictor values into a dataframe
    feature_vector_df = pd.DataFrame(scaled, columns=test_data.columns)
    #feature_vector_df = pd.DataFrame(scaled, columns=X_train.columns)
    #feature_vector_df = feature_vector_df.drop('time', axis=1)
    
    feature_vector_df = feature_vector_df.fillna(feature_vector_df['Valencia_pressure'].mean())
    #feature_vector_df = feature_vector_df.fillna(method='ffill')
    
    predict_vector = feature_vector_df#.iloc[:, :42]

    # ------------------------------------------------------------------------

    return predict_vector

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))


""" You may use this section (above the make_prediction function) of the python script to implement 
    any auxiliary functions required to process your model's artifacts.
"""

def make_prediction(data, model):
    """Prepare request data for model prediction.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standardisation.
    #return prediction[0].tolist()
    return prediction.tolist()
