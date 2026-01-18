import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import gc
import os 

# --- Configuration ---
PROJECT_FOLDER = './data/'

def create_features(df):
    """Crée toutes les features nécessaires à partir du DataFrame complet."""
    print("  - Création des features temporelles...")
    df['date'] = pd.to_datetime(df['date'])
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_year'] = df['date'].dt.dayofyear
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    
    print("  - Création des Lag et Rolling Window Features...")
    df = df.sort_values(by=['station', 'date'])
    
    lags = [1, 2, 7, 14, 28]
    for lag in lags:
        df[f'y_lag_{lag}'] = df.groupby('station')['y'].shift(lag)

    windows = [7, 14, 28]
    for window in windows:
        shifted_y = df.groupby('station')['y'].shift(1)
        df[f'y_rolling_mean_{window}'] = shifted_y.rolling(window).mean()
        df[f'y_rolling_std_{window}'] = shifted_y.rolling(window).std()

    return df

# --- Script Principal ---
if __name__ == "__main__":
    print("1. Chargement et fusion des données...")
    try:
        x_train_df = pd.read_csv(os.path.join(PROJECT_FOLDER, 'x_train.csv'))
        y_train_df = pd.read_csv(os.path.join(PROJECT_FOLDER, 'y_train.csv'))
        x_test_df = pd.read_csv(os.path.join(PROJECT_FOLDER, 'x_test.csv'))
    except FileNotFoundError:
        print(f"ERREUR: Fichiers non trouvés dans '{PROJECT_FOLDER}'.")
        raise

    
    # Fusionner x_train et y_train
    x_train_df['index'] = pd.to_datetime(x_train_df['date']).dt.strftime('%Y-%m-%d') + '_' + x_train_df['station']
    df_train = pd.merge(x_train_df, y_train_df, on='index')

    # Garder uniquement les colonnes communes et essentielles avant de concaténer
    common_cols = ['date', 'station', 'job', 'ferie', 'vacances', 'y']
    df_train = df_train[common_cols]
    x_test_df = x_test_df[['date', 'station', 'job', 'ferie', 'vacances']] 
    df_full = pd.concat([df_train, x_test_df.assign(y=np.nan)], sort=False).reset_index(drop=True)

    # Encodage de la station
    le = LabelEncoder()
    df_full['station_encoded'] = le.fit_transform(df_full['station'])


    del x_train_df, y_train_df
    gc.collect()

    print("\n2. Création des features...")
    df_full = create_features(df_full)

    print("\n3. Séparation des données...")
    train_df = df_full[df_full['y'].notna()].copy()
    test_df = df_full[df_full['y'].isna()].copy()
    
    # Éviter la division par zéro dans la MAPE : on ne s'entraîne pas sur les jours à 0 validation.
    # C'est une petite fraction des données mais stabilise l'entraînement MAPE.
    train_df = train_df[train_df['y'] > 0].copy()

    features = [col for col in train_df.columns if col not in ['date', 'station', 'y']]
    target = 'y'

    print(f"Nombre de features utilisées : {len(features)}")

    # Utiliser les données de 2022 comme validation pour simuler le challenge
    X_train = train_df[train_df['year'] < 2022][features]
    y_train = train_df[train_df['year'] < 2022][target]
    X_val = train_df[train_df['year'] == 2022][features]
    y_val = train_df[train_df['year'] == 2022][target]
    
    X_test = test_df[features]

    # Convertir 'station' en type 'category' pour LightGBM
    X_train['station_encoded'] = X_train['station_encoded'].astype('category')
    X_val['station_encoded'] = X_val['station_encoded'].astype('category')
    X_test['station_encoded'] = X_test['station_encoded'].astype('category')

    print("\n4. Entraînement du modèle LightGBM (optimisé pour MAPE)...")
    lgb_params = {
        'objective': 'mape',  
        'metric': 'mape',     
        'n_estimators': 2000,
        'learning_rate': 0.02,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'num_leaves': 31,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42,
    }
    
    model = lgb.LGBMRegressor(**lgb_params)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='mape',
        callbacks=[lgb.early_stopping(100, verbose=True)]
    )

    print("\n5. Génération des prédictions finales...")
    predictions = model.predict(X_test)
    final_predictions = np.maximum(0, predictions).round().astype(int)

    print("\n6. Création du fichier de soumission...")
    submission_df = pd.DataFrame({
        'index': pd.to_datetime(test_df['date']).dt.strftime('%Y-%m-%d') + '_' + test_df['station'],
        'y': final_predictions
    })
    
    submission_filename = 'submission_lightgbm_mape_optimized.csv'
    submission_df.to_csv(submission_filename, index=False)

    print(f"\nFichier de soumission '{submission_filename}' créé avec succès !")
    print(submission_df.head())