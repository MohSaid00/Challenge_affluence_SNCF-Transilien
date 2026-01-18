import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import os

# --- Configuration 
PROJECT_FOLDER = './data/' 

# Paramètres du modèle et des séquences
SEQUENCE_LENGTH = 14  # On utilise les 14 derniers jours pour prédire le suivant (ni)
LSTM_UNITS = 100
EPOCHS = 20
BATCH_SIZE = 256

# --- Fonctions ---

def feature_engineering(df):
    """Crée des features temporelles à partir de la colonne 'date'."""
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_year'] = df['date'].dt.dayofyear
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    return df

def create_sequences_multivariate(df, stations_list, ni, feature_cols, target_col_index=0):
    """Crée des séquences (X, y) pour un dataset multivarié, gare par gare."""
    X, y = [], []
    df_sorted = df.sort_values(by=['station', 'date'])

    for station in stations_list:
        station_df = df_sorted[df_sorted['station'] == station].copy()
        if len(station_df) < ni + 1:
            continue
        data = station_df[feature_cols].values
        for i in range(len(data) - ni):
            X.append(data[i: i + ni])
            y.append(data[i + ni, target_col_index])
    return np.array(X), np.array(y)

def build_sncf_model(ni, n_features, lstm_units=LSTM_UNITS):
    """Construit un modèle LSTM robuste pour le challenge SNCF."""
    model = Sequential([
        LSTM(lstm_units, activation='tanh', input_shape=(ni, n_features), return_sequences=True),
        Dropout(0.2),
        LSTM(lstm_units // 2, activation='tanh'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='mse',
                  metrics=['mean_absolute_percentage_error'])
    return model

if __name__ == "__main__":
    
    # 1. Chargement et préparation des données
    print("1. Chargement des données...")
    try:
        x_train_df = pd.read_csv(os.path.join(PROJECT_FOLDER, 'x_train.csv'), parse_dates=['date'])
        y_train_df = pd.read_csv(os.path.join(PROJECT_FOLDER, 'y_train.csv'))
        x_test_df = pd.read_csv(os.path.join(PROJECT_FOLDER, 'x_test.csv'), parse_dates=['date'])
        sample_submission_df = pd.read_csv(os.path.join(PROJECT_FOLDER, 'sample_submission.csv'))
    except FileNotFoundError as e:
        print(f"Erreur: Fichier non trouvé. Assurez-vous que les fichiers CSV sont dans le dossier: {PROJECT_FOLDER}")
        raise e

    x_train_df['index'] = x_train_df['date'].dt.strftime('%Y-%m-%d') + '_' + x_train_df['station']
    df_train = pd.merge(x_train_df, y_train_df, on='index')
    df_train.drop(columns=['index'], inplace=True)

    # 2. Feature Engineering
    print("2. Création des features...")
    df_train = feature_engineering(df_train)
    x_test_df = feature_engineering(x_test_df)

    # 3. Normalisation
    print("3. Normalisation des données...")
    feature_cols = ['y', 'job', 'ferie', 'vacances', 'day_of_week', 'day_of_year', 'month', 'is_weekend']
    target_col = 'y'
    
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    df_train[feature_cols] = feature_scaler.fit_transform(df_train[feature_cols])
    target_scaler.fit(df_train[[target_col]]) # Scaler spécifique pour la cible

    # 4. Création des séquences d'entraînement
    print("4. Création des séquences d'entraînement...")
    y_col_index = feature_cols.index(target_col)
    stations_to_train = df_train['station'].unique()
    X_train_seq, y_train_seq = create_sequences_multivariate(df_train, stations_to_train, SEQUENCE_LENGTH, feature_cols, y_col_index)
    
    print(f"Forme des données d'entrée X: {X_train_seq.shape}")
    print(f"Forme des données de sortie y: {y_train_seq.shape}")

    # 5. Construction et entraînement du modèle
    print("5. Construction et entraînement du modèle LSTM...")
    n_features = X_train_seq.shape[2]
    model = build_sncf_model(SEQUENCE_LENGTH, n_features)
    model.summary()
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    history = model.fit(
        X_train_seq, y_train_seq,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )

    # 6. Préparation des données de test pour la prédiction
    print("6. Préparation des données de test...")
    last_days_from_train = df_train.groupby('station').tail(SEQUENCE_LENGTH)
    x_test_with_dummies = x_test_df.copy()
    x_test_with_dummies['y'] = 0 # Colonne factice pour la compatibilité
    
    combined_df = pd.concat([last_days_from_train, x_test_with_dummies], ignore_index=True)
    combined_df = combined_df.sort_values(by=['station', 'date'])
    
    # Appliquer la même normalisation
    combined_df[feature_cols] = feature_scaler.transform(combined_df[feature_cols])

    # 7. Génération des prédictions
    print("7. Génération des prédictions finales...")
    all_predictions_scaled = []
    all_corresponding_ids = []
    
    x_test_df['index'] = x_test_df['date'].dt.strftime('%Y-%m-%d') + '_' + x_test_df['station']

    for station in x_test_df['station'].unique():
        station_data_combined = combined_df[combined_df['station'] == station]
        
        if len(station_data_combined) < SEQUENCE_LENGTH:
            continue
        
        # Créer les séquences d'entrée pour la prédiction
        station_features = station_data_combined[feature_cols].values
        input_sequences_test = []
        for i in range(len(station_data_combined) - SEQUENCE_LENGTH):
             input_sequences_test.append(station_features[i : i + SEQUENCE_LENGTH])

        if not input_sequences_test:
            continue
        
        X_pred = np.array(input_sequences_test)
        predictions_scaled = model.predict(X_pred, verbose=0)
        
        station_ids = x_test_df[x_test_df['station'] == station]['index'].values
        
        all_predictions_scaled.extend(predictions_scaled.flatten())
        all_corresponding_ids.extend(station_ids)

    print(f"{len(all_predictions_scaled)} prédictions ont été générées.")

    # 8. Dés-normalisation et création du fichier de soumission
    print("8. Création du fichier de soumission...")
    predictions_original_scale = target_scaler.inverse_transform(np.array(all_predictions_scaled).reshape(-1, 1))
    final_predictions = np.maximum(0, predictions_original_scale.flatten().round()).astype(int)

    submission_df = pd.DataFrame({
        'index': all_corresponding_ids,
        'y': final_predictions
    })
    
    # Fusionner pour garantir le bon format et l'ordre
    final_submission_df = pd.merge(sample_submission_df[['index']], submission_df, on='index', how='left')
    final_submission_df['y'].fillna(0, inplace=True)
    final_submission_df['y'] = final_submission_df['y'].astype(int)

    submission_filename = 'submission_lstm.csv'
    final_submission_df.to_csv(submission_filename, index=False)
    
    print(f"\nFichier de soumission '{submission_filename}' créé avec succès !")
    print(final_submission_df.head())