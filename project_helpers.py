import numpy as np
import pandas as pd
import random
import math


def get_users_ID(line):
    '''Get the user_ID in correct surprise format'''
    row, col = line.split("_")
    row = row.replace("r", "")
    return int(row)

def get_movie_ID(line):
    '''get the movie_ID in the correct surprise format'''
    row, col = line.split("_")
    col = col.replace("c", "")
    return int(col)

def df_to_surprise(data):
    '''Change the dataframe from the user_ID_movie_ID_rate format to the userID;itemID;rating format'''
    data['userID'] = data['Id'].apply(get_users_ID)
    data['itemID'] = data['Id'].apply(get_movie_ID)
    data = data.drop('Id', axis=1)
    data = data.rename(columns={'Prediction':'rating'})[['userID','itemID','rating']]
    return data

def global_mean(df):
    '''Return the global dataset mean'''
    return df.rating.mean()

def user_mean(df):
    '''Return a serie where each user is associated with his mean'''
    return df.groupby('userID').rating.mean()

def movie_mean(df):
    '''Return a serie where each movie is associated with its mean'''
    return df.groupby('itemID').rating.mean()