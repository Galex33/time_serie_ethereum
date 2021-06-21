import pandas as pd

def read_copy_csv(csv, parse_dates_bool=False, index_col= None ):
    data = pd.read_csv(csv, parse_dates=parse_dates_bool, index_col = index_col)
    df = data.copy()
    return df

def df_info(df):
        len_df = len(df)
        all_columns = len(df.columns)

        print(f"""
        Longueur du dataset : {len_df} enregistrements
        Nombre de colonnes : {all_columns}
        """)

        echantillonColonnes = []
        for i in df.columns:
            listcolumn = str(list(df[i].head(5)))
            echantillonColonnes.append(listcolumn)
       
        pd.set_option("max_rows", None)
        obs = pd.DataFrame({'type': list(df.dtypes),
        'Echantillon': echantillonColonnes,
        "% de valeurs nulles":
        round(df.isna().sum() / len_df * 100, 2),
        'Nbr L dupliquées' : (df.duplicated()).sum(),
        'Nbr V unique' : df.nunique()
        })
        return obs

def mask_value(df, col, value): 
    df_mask= df[col] == value
    filtered_df = df[df_mask]
    return filtered_df

def new_df(df, columns):
    df1 = df[columns]
    new_names = {
        "Date": "ds", 
        "Opening_Price_ETH": "y",
    }
    df1 = df1.rename(columns=new_names)
    return df1

