import pandas as pd
from src.logger import logger

def load_data(train_path,test_path):
    logger.info('Loading datasets...')
    train=pd.read_csv(train_path)
    test=pd.read_csv(test_path)
    return train,test

def convert_datetime(df):
    df['DateTime']=pd.to_datetime(df['DateTime'])
    return df

def remove_duplicates(df):
    before=len(df)
    df=df.drop_duplicates()
    logger.info(f'Removed {before-len(df)} duplicate rows.')
    return df

def preprocess(train,test):
    train=convert_datetime(train)
    test=convert_datetime(test)
    train=remove_duplicates(train)
    return train,test
