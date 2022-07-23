import numpy as np
import pandas as pd

np.random.seed(42)
pd_RANDOM_STATE = 42

def split_data(filename: str = 'labeled_data.csv'):
    '''Split the raw data set into train, validation and test set. Export the raw dataframe into csv files. Default ratio: 8:1:1.'''
    df = pd.read_csv(f'data/{filename}', index_col=0)
    df.sample(frac=1, random_state=pd_RANDOM_STATE).reset_index(drop=True)

    factor = len(df) // 10
    train_df = df[0:(8*factor)]
    val_df = df[(8*factor):(9*factor)]
    test_df = df[(9*factor):]

    train_df.to_csv('data/train_data.csv')
    val_df.to_csv('data/val_data.csv')
    test_df.to_csv('data/test_data.csv')

    return

def main():
    split_data('labeled_data.csv')

if __name__ == "__main__":
    main()