import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

def engineer_features(X_df, y):
    X = X_df.copy()
    if 'order_date' in X:
        X['order_date'] = pd.to_datetime(X['order_date'], errors='coerce')
        X['month'] = X['order_date'].dt.month
        X['weekday'] = X['order_date'].dt.weekday
        X.drop(columns=['order_date'], inplace=True)

    X = pd.get_dummies(X, drop_first=True)

    y = pd.Series(y)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    if len(set(y_enc)) > 1:
        X_scaled = StandardScaler().fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        sm = SMOTE()
        X_res, y_res = sm.fit_resample(X_scaled, y_enc)
    else:
        X_res, y_res = X, y_enc

    return X_res, y_res, le
