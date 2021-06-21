import pandas as pd



def drop_rename_col(df, ls_drop, ls_rename):
    df.drop(columns=ls_drop, inplace=True)
    df.columns = ls_rename
    return df

def del_dupicated(df):
    df = df[~df.duplicated()]
    return df

def formatage_date(df, col_date):
    df[col_date] = df[col_date].apply(lambda x:x.replace(' 04:00', ''))
    df[col_date] = pd.to_datetime(df[col_date], dayfirst=True )
    return df

def cleaning(df):
    ethereum_day = drop_rename_col(df, ['Unix Timestamp', 'Symbol'],['Date', 'Opening_Price_ETH', 'Highest_rice_ETH', 'Lowest_Price_ETH', 'Lowest_Price_ETH', 'Vol_ETH'])
    ethereum_day = del_dupicated(ethereum_day)
    ethereum_day = formatage_date(ethereum_day, 'Date')
    return ethereum_day

