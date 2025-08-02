import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    data = pd.read_csv(file_path, encoding='ISO-8859-1')
    print(f"Loaded data shape: {data.shape}")
    return data

def clean_data(df):
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%d-%m-%Y %H:%M')
    df = df.dropna(subset=['CustomerID'])
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]
    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
    df = df[df['TotalAmount'] <= df['TotalAmount'].quantile(0.99)]
    df['CustomerID'] = df['CustomerID'].astype(int)
    print(f"Cleaned data shape: {df.shape}")
    return df

def create_features(df):
    latest_date = df['InvoiceDate'].max()
    
    features = df.groupby('CustomerID').agg({
        'InvoiceDate': ['count', 'max'],
        'TotalAmount': ['sum', 'mean'],
        'Quantity': 'sum',
        'InvoiceNo': 'nunique'
    })
    
    features.columns = ['Frequency', 'LastPurchase', 'TotalSpent', 'AvgOrderValue', 'TotalQuantity', 'NumSessions']
    features['Recency'] = (latest_date - features['LastPurchase']).dt.days
    features = features.drop('LastPurchase', axis=1)
    
    features = features[features['Frequency'] > 1]
    print(f"Features shape: {features.shape}")
    return features

def scale_features(features):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    scaled_df = pd.DataFrame(scaled, columns=features.columns, index=features.index)
    return scaled_df, scaler 