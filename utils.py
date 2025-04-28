import pandas as pd

def print_df_status(df:pd.DataFrame)->None:
    print(f'Current version of the df has the following columns\n{df.columns}')

def save_new_df(df:pd.DataFrame)-> None:
    print('Saving current version of the df. \n The following columns are present: ')
    print(df.columns)
    print('Saving to current_features.csv')
    df.to_csv("./current_features.csv", index=False)
    
    