import pandas as pd
import numpy as np
import time
import warnings

# used to read the dataset 
def get_dataframe(dataset):
    df = pd.read_csv(dataset)
    # gets a list of column names
    column_names = df.columns.tolist()

    for column in column_names:
        df[column] = pd.to_numeric(df[column], errors='coerce').fillna(-1)

    df = df.round(decimals=5)
    return df

# implements mean imputation
def mean_imputation(df):

    # gets a list of column names
    column_names = df.columns.tolist()

    # for every column, find the average of that column and set every '?' to the mean of that column
    for column in column_names:
        df[column] = pd.to_numeric(df[column])
        mean = df[column].mean()
        df[column] = df[column].replace(-1, mean)

    # return dataframe
    rounded_df = df.round(5)
    return df

def calculate_manhattan_distances(df):
    # make a square matrix to store manhattan distances
    number_of_objects = df.shape[0]
    columns = df.shape[1]
    distances = np.zeros((number_of_objects,columns))

    for i in range(number_of_objects):
        for j in range(i+1, columns):
            least_distance = float('inf')
            for k in range(number_of_objects):
                distance = 0
                for l in range(columns):
                    x = df[i, j]
                    y = df[k, l]
                    if x ==-1 or y ==-1:
                        distance += 1
                    else:
                        distance += abs(x - y)
                if distance < least_distance:
                    least_distance = distance
            distances[i, j] = distance
            distances[j, i] = distance
    return distances


# implements hot deck imputation
def hot_deck_imputation(df):
    # create a deep copy of the dataframe to not impute missing values with imputed values
    new_df = df.copy(deep=True)
    columns = new_df.columns.tolist()
    df = df.to_numpy()
    df.astype(float)

    distances = calculate_manhattan_distances(df)
    distances_df = pd.DataFrame(distances, columns=columns)
    # go through every value and find '?'
    for index, row in new_df.iterrows():
        for column in new_df.columns.tolist():
            if row[column] == -1:
                new_df.at[index, column] = distances_df.at[index, column]
                
    rounded_df = new_df.round(5)
    return rounded_df

def mae(missing_df, imputated_df, complete_df):
    missing_data_count = 0
    sum = 0
    for index, row in missing_df.iterrows():
        for column in missing_df.columns.tolist():
            if row[column] == -1:
                missing_data_count = missing_data_count + 1
                sum += abs(imputated_df.at[index, column] - complete_df.at[index, column])
    mae_value = sum / missing_data_count
    mae_value = round(mae_value, 4)
    return mae_value

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    df01 = get_dataframe('dataset_missing01.csv')
    df10 = get_dataframe('dataset_missing10.csv')
    dfcp = get_dataframe('dataset_complete.csv')

    start_time = time.time()*1000
    mi01 = mean_imputation(df01)
    runtime_01_mean = round(time.time()*1000 - start_time)
   
    start_time = time.time()*1000
    mi10 = mean_imputation(df10)
    runtime_10_mean = round(time.time()*1000 - start_time)
    
    start_time = time.time()*1000
    hd01 = hot_deck_imputation(df01)
    runtime_01_hd = round(time.time()*1000 - start_time)
    
    start_time = time.time()*1000
    hd10 = hot_deck_imputation(df10)
    runtime_10_hd = round(time.time()*1000 - start_time)

    mi01.to_csv('V00924744_missing01_imputed_mean.csv')
    mi10.to_csv('V00924744_ missing10_imputed_mean.csv')
    hd01.to_csv('V00924744_missing01_imputed_hd.csv')
    hd10.to_csv('V00924744_missing10_imputed_hd.csv')
    
    dfvmi01 = get_dataframe('V00924744_missing01_imputed_mean.csv')
    dfvmi10 = get_dataframe('V00924744_ missing10_imputed_mean.csv')
    dfvhd01 = get_dataframe('V00924744_missing01_imputed_hd.csv')
    dfvhd10 = get_dataframe('V00924744_missing10_imputed_hd.csv')

    df01 = get_dataframe('dataset_missing01.csv')
    df10 = get_dataframe('dataset_missing10.csv')
    dfcp = get_dataframe('dataset_complete.csv')

    vmi01 = mae(df01, dfvmi01, dfcp)

    vmi10 = mae(df10, dfvmi10, dfcp)

    vhd01 = mae(df01, dfvhd01, dfcp)

    vhd10 = mae(df10, dfvhd10, dfcp)
    
    print(f"MAE_01_mean = {vmi01}")
    print(f"Runtime_01_mean = {runtime_01_mean}")
    print(f"MAE_01_hd = {vhd01}")
    print(f"Runtime_01_hd = {runtime_01_hd}")
    print(f"MAE_10_mean = {vmi10}")
    print(f"Runtime_10_mean = {runtime_10_mean}")
    print(f"MAE_10_hd = {vhd10}")
    print(f"Runtime_10_hd = {runtime_10_hd}")

    