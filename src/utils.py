import pandas as pd
from scipy.sparse import hstack

def load_data(path):
    df = pd.read_csv(path)
    return df


def preprocess_text(df):
    df['journal_text'] = df['journal_text'].fillna("")
    return df


def get_meta_features(df):
    meta_cols = ['sleep_hours', 'energy_level', 'stress_level', 'duration_min']
    return df[meta_cols].fillna(0)


def combine_features(X_text, X_meta):
    return hstack([X_text, X_meta])


def handle_missing(df):
    return df.fillna(0)


def clean_time(df):
    if 'time_of_day' in df.columns:
        df['time_of_day'] = df['time_of_day'].fillna("unknown")
    return df
