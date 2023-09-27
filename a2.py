import pandas as pd
import numpy as np

def get_numpy(dataset):
    df = pd.read_csv(dataset)
    df.to_numpy()
    return df

if __name__ == '__main__':
    a = get_numpy('dataset_missing10.csv')

    print(a)