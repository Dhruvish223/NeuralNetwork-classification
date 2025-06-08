import pandas as pd

def enforce_column_types(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')  # convert object to numeric, else NaN
            except Exception:
                pass
    df.fillna(0, inplace=True)  # Fill NaNs (both coerced and original) with zero
    return df

def clean_data(df):
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')  # normalize col names
    df.drop_duplicates(inplace=True)
    df.dropna(axis=1, thresh=int(0.7 * len(df)), inplace=True)  # drop cols with >30% nulls

    # Fill categorical missing values with mode
    for c in df.select_dtypes('object'):
        df[c].fillna(df[c].mode()[0], inplace=True)

    # Fill numeric missing values with median
    for c in df.select_dtypes(['int64', 'float64']):
        df[c].fillna(df[c].median(), inplace=True)

    df = enforce_column_types(df)  # enforce numeric conversion after filling

    return df
