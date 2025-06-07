from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def engineer_features(df):
    df = df.copy()
    
    # Example: Convert Order Date to features
    if 'order_date' in df.columns:
        df['order_date'] = pd.to_datetime(df['order_date'])
        df['order_month'] = df['order_date'].dt.month
        df['order_day'] = df['order_date'].dt.dayofweek

    # Encode categorical variables
    cat_cols = df.select_dtypes(include='object').columns
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    # Assume last column is target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    return X, y, le
