import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data():
    df = pd.read_csv('dataset.csv', delimiter=';')
    df['ReturnAfterPPC'] = df['ReturnAfterPPC'].str.replace(',', '.').astype(float)
    df['Price'] = df['Price'].str.replace(',', '.').astype(float)
    
    # Remove extreme outliers from ReturnAfterPPC
    q_low = df['ReturnAfterPPC'].quantile(0.01)
    q_high = df['ReturnAfterPPC'].quantile(0.99)
    df_filtered = df[(df['ReturnAfterPPC'] >= q_low) & (df['ReturnAfterPPC'] <= q_high)]
    
    scaler = MinMaxScaler()
    features_to_scale = ['ReturnAfterPPC', 'Price']
    df_filtered[features_to_scale] = scaler.fit_transform(df_filtered[features_to_scale])
    
    X = df_filtered[['ReturnAfterPPC']]
    y = df_filtered['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return df_filtered, X_train, X_test, y_train, y_test
