import pandas as pd

def clean_data(df):
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    df.drop_duplicates(inplace=True)
    df.dropna(axis=1, thresh=int(0.7 * len(df)), inplace=True)
    for c in df.select_dtypes('object'): df[c].fillna(df[c].mode()[0], inplace=True)
    for c in df.select_dtypes(['int64','float64']): df[c].fillna(df[c].median(), inplace=True)
    return df
