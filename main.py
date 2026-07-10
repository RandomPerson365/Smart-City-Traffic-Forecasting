from config.config import RAW_DATA
from src.preprocessing import load_data, preprocess
from src.utils import create_directories

def main():
    create_directories()
    train,test=load_data(RAW_DATA/'train.csv',RAW_DATA/'test.csv')
    train,test=preprocess(train,test)
    print(train.head())

if __name__=='__main__':
    main()
