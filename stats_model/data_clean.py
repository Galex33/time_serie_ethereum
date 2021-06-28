import pandas as pd
import numpy as np

def drop_rename_col(df, ls_drop, ls_rename):
    df.drop(columns=ls_drop, inplace=True)
    df.columns = ls_rename
    return df

def formatage_date(df, col_date, path_save_csv, encoding):
    df[col_date] = df[col_date].apply(lambda x:x.replace(' 04:00', ''))
    df[col_date] = pd.to_datetime(df[col_date], dayfirst=True )
    df = df.set_index(col_date)
    df = df.sort_index(axis = 0, ascending = True)
    df.to_csv(path_save_csv, encoding=encoding)
    return df

def cleaning(df, ls_drop, ls_rename, col_date, path_save_csv, encoding):
    df = drop_rename_col(df, ls_drop, ls_rename)
    df = formatage_date(df, col_date, path_save_csv, encoding)
    return df

