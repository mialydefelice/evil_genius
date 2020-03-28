import numpy as np
import pandas as pd
import re


def rename_df_cols(df):
    """
    The purpose of this function is to take the underscore and numbering off column names if they exist.
    These exist due to how treatment pathways is structured.
    """
    new_col_names = {}
    for col in df.columns:
        match = re.search('_\d', col)
        if match:
            last_index = match.span()[0]
            new_col_names[col] = col[0:last_index]
    renamed_df = df.rename(columns=new_col_names)
    return renamed_df

def extract_binary_data(df):
    """
    Returns the df with only columns that are represented by binary values. Pulls out columns that I already know arent binary. 
    """
    return df.drop(['Source Person ID', 'Datepointer', 'Birthdate', 'age_at_treatment'], axis = 1)

