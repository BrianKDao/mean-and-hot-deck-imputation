import pandas as pd
import numpy as np
import time

# used to read the dataset 
def get_dataframe(dataset):
    df = pd.read_csv(dataset)
    return df

# implements mean imputation
def mean_imputation(dataset):
    # gets the dataframe
    df = get_dataframe(dataset)

    # gets a list of column names
    column_names = df.columns.tolist()

    # for every column, find the average of that column and set every '?' to the mean of that column
    for column in column_names:
        df[column] = pd.to_numeric(df[column], errors='coerce')
        mean = df[column].mean()
        df[column] = df[column].fillna(mean)

    # return dataframe
    return df

# def impute_missing_value(df, current_index, current_row):

#     # set the nearest object to infinity so every other object is less than it, to ensure first case is less than nearest object
#     nearest_object = float('inf')
#     # iterate through all the rows
#     for index, row in df.iterrows():
#         # initialize sum and get the items in the row
#         sum = 0

#         # don't compare object to itself
#         if index is not current_index:
#             # for each column, find the abs value of them added together, unless either of the values are a '?', then add 1
#             for column in df.columns.tolist():

#                 input_row = df.at[current_index, column]
#                 loop_row = df.at[index, column]

#                 if loop_row == '?' or input_row == '?':
#                     sum += 1
#                 else:
#                     sum += abs(float(loop_row) + float(input_row))

#         # if the sum is less than the current nearest object, change it to this one
#         if sum < nearest_object:
#             nearest_object = sum

#     # return the imputed value
#     return nearest_object
    

def calculate_manhattan_distances(df):
    # make a square matrix to store manhattan distances
    number_of_objects = df.shape[0]
    distances = np.zeros((number_of_objects,number_of_objects))

    for i in range(number_of_objects):
        for j in range(i+1, number_of_objects):
            distance = 0
            for column in df.columns:
                x = df.at[i, column]
                y = df.at[j, column]
                if x =='?' or y =='?':
                    distance += 1
                else:
                    distance += abs(float(x) - float(y))
            
            distances[i, j] = distance
            distances[j, i] = distance
        print("Still Running")
    return distances
# implements hot deck imputation
def hot_deck_imputation(dataset):
    # gets the dataframe
    df = get_dataframe(dataset)
    number_of_objects = df.shape[0]

    distances = calculate_manhattan_distances(df)
    # create a deep copy of the dataframe to not impute missing values with imputed values
    new_df = df.copy(deep=True)

    # go through every value and find '?'
    for i in range(n):
        for column_name, value in df.iloc[i].items():
          # for every '?', run the impute missing value function and add that value to the new dataframe
            nearest_index = np.argmin(distances[i, :])
            imputed_value = df.at[nearest_index, column_name]
            new_df.at[i, column_name] = imputed_value
        print("Still Running")

    return new_df



if __name__ == '__main__':
    start_time = time.time()*1000
    print(hot_deck_imputation('dataset_missing10.csv'))
    print(time.time()*1000 - start_time)
    
    