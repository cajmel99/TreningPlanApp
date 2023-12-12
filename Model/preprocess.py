import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

translation_dict = {
    'Typ aktywności': 'Activity Type',
    'Data': 'Date',
    'Ulubiony': 'Favorite',
    'Tytuł': 'Title',
    'Dystans': 'Distance',
    'Kalorie': 'Calories',
    'Czas': 'Time',
    'Średnie tętno': 'Avg HR',
    'Maksymalne tętno': 'Max HR',
    'Aerobowy TE': 'Areobic TE',
    'Średni rytm biegu': 'Avg Run Cadence',
    'Maksymalny rytm biegu': 'Max Run Cadence',
    'Średnie tempo': 'Avg Pace',
    'Najlepsze tempo': 'Best Pace',
    'Całkowity wznios': 'Total Ascent',
    'Całkowity spadek': 'Total Descent',
    'Średnia długość kroku': 'Avg Stride Length',
    'Średnie odchylenie do długości': 'Avg Vertical Ratio',
    'Średnie odchylenie pionowe': 'Avg Vertical Oscillation',
    'Średni czas kontaktu z podłożem': 'Avg Ground Contact Time',
    'Średni GAP': 'Avg Ground Angular Power',
    'Normalized Power® (NP®)': 'Normalized Power (NP)',
    'Training Stress Score® (TSS®)': 'Training Stress Score®',
    'Średnia moc': 'Avg Power',
    'Maksymalna moc': 'Max Power',
    'Trudność': 'Grit',
    'Płynność': 'Flow',
    'Średni Swolf': 'Avg. Swolf',
    'Średnie tempo ruchów': 'Avg Stroke Rate',
    'Razem powtórzeń': 'Total Reps',
    'Czas nurkowania': 'Dive Time',
    'Minimalna temperatura': 'Min Temp',
    'Przerwa powierzchniowa': 'Surface Interval',
    'Dekompresja': 'Decompression',
    'Czas najlepszego okrążenia': 'Best Lap Time',
    'Liczba okrążeń': 'Number of Laps',
    'Maksymalna temperatura': 'Max Temp',
    'Czas ruchu': 'Moving Time',
    'Upłynęło czasu': 'Elapsed Time',
    'Minimalna wysokość': 'Min Elevation',
    'Maksymalna wysokość': 'Max Elevation',
    'Średnia częstotliwość oddechu': 'Avg Breath',
    'Minimalna częstotliwość oddechu': 'Min Breath', 
    'Maksymalna częstotliwość oddechu': 'Max Breath'
}
z_score_threshold = 3  # As the rule from thumb say 

"""Utilty functions"""
def convert_time_to_seconds(time_str):
        parts = time_str.split(':')
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(float(parts[2]))
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(float(parts[1]))
        return time_str  

# Function to identify and label outliers using Z-scores
def detect_outliers(column):
    mean = column.mean()
    std = column.std()
    z_scores = (column - mean) / std
    return abs(z_scores) > z_score_threshold

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    #Reverse the order (to have data from first date of training to last date of training)
    df = df[::-1].reset_index(drop=True)
    df.rename(columns=translation_dict, inplace=True)
    columns_to_drop = [
        'Activity Type', 'Favorite', 'Title', 'Calories', 'Dive Time', 'Avg Vertical Ratio', 
        'Avg Vertical Oscillation', 'Avg Ground Contact Time', 'Training Stress Score®', 'Avg Power',
        'Max Power', 'Grit', 'Flow', 'Avg. Swolf', 'Avg Stroke Rate',
        'Total Reps', 'Dive Time', 'Min Temp', 'Surface Interval', 'Time', 'Decompression', 
        'Best Lap Time', 'Number of Laps', 'Max Temp', 'Elapsed Time', 'Min Elevation',
        'Max Elevation', 'Moving Time', 'Avg Run Cadence', 'Avg HR', 'Best Pace'
    ]
    other_columns = ['Areobic TE','Avg Ground Angular Power', 'Normalized Power (NP)', 'Avg Breath', 
                     'Min Breath', 'Max Breath']

    df = df.drop(columns=columns_to_drop, errors='ignore')

    df_columns  = df.columns
    for i in df_columns:
        for j in other_columns:
            if i == j:
                df = df.drop(i, axis=1)
    
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = df['Date'].astype(int) // 10**9

    columns_to_convert = [ 'Max Run Cadence', 'Total Ascent', 'Total Descent'] 
    for col in columns_to_convert:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    time_columns = ['Avg Pace']
    for col in time_columns:
        df[col] = df[col].apply(convert_time_to_seconds)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    numerical_columns = df.select_dtypes(include='number')
    outliers = numerical_columns.apply(detect_outliers)

    outliers_rows = outliers[outliers.any(axis=1)]
    df['Has_Outlier'] = df.index.isin(outliers_rows.index)
    df_clear = df[df['Has_Outlier'] == False]
    df_clear = df_clear.drop('Has_Outlier', axis=1)
    df = df_clear
    df = df.fillna(df.median())

    return df

def split_data(df):
    X = df.drop('Avg Pace', axis =1)
    y = df['Avg Pace']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, shuffle=False)

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)

    return X_train, X_test, X_val, y_train, y_test, y_val, scaler
