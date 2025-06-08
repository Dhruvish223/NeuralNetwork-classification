import pandas as pd

def is_numeric_column(series: pd.Series, threshold=0.9) -> bool:
    """
    Checks if a column is mostly numeric (percentage of parsable numbers > threshold).
    """
    num_parsable = pd.to_numeric(series, errors='coerce').notna().sum()
    total = len(series)
    return (num_parsable / total) >= threshold

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    df.drop_duplicates(inplace=True)
    df.dropna(axis=1, thresh=int(0.7 * len(df)), inplace=True)  # drop columns with >30% nulls

    for col in df.columns:
        if df[col].dtype == 'object':
            if is_numeric_column(df[col]):
                # Convert to numeric if mostly numeric strings
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Fill NaNs with median for numeric columns
                df[col].fillna(df[col].median(), inplace=True)
            else:
                # Otherwise treat as categorical/text: fill NaNs with mode
                df[col].fillna(df[col].mode()[0], inplace=True)
        elif pd.api.types.is_numeric_dtype(df[col]):
            # For numeric columns: fill NaNs with median
            df[col].fillna(df[col].median(), inplace=True)
        else:
            # For other types, fill NaNs with mode (safe fallback)
            df[col].fillna(df[col].mode()[0], inplace=True)

    return df
