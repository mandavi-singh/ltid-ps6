"""
Model Training Script for Agriculture Dataset
--------------------------------------------

This script performs preprocessing and regression modelling on the Agriculture
and Farming dataset. It encodes categorical variables, splits the data into
training and testing sets, trains multiple regression models, evaluates them
and saves the best-performing model along with the preprocessor pipeline for
use in deployment (e.g., in a Streamlit app).  It also contains an example
function for training an LSTM model using TensorFlow/Keras.  The LSTM
function is provided for completeness but is not executed by default because
TensorFlow may not be installed in all environments.

Usage:
    python model_training.py

Outputs:
    - preprocessor.pkl: serialized preprocessing pipeline.
    - best_model.pkl: serialized best regression model (RandomForest, XGB or
      GradientBoosting, depending on evaluation).
    - results.csv: summary of cross‑validation and test performance for each
      model.

"""

import os
import joblib
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

# Path constants
DATA_PATH = os.path.join(os.path.dirname(__file__), 'agriculture_dataset.csv')
PREPROCESSOR_PATH = os.path.join(os.path.dirname(__file__), 'preprocessor.pkl')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best_model.pkl')
RESULTS_PATH = os.path.join(os.path.dirname(__file__), 'results.csv')


def load_data(path: str) -> pd.DataFrame:
    """Load the dataset from a CSV file."""
    return pd.read_csv(path)


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Construct a preprocessing pipeline for the dataset.

    Categorical columns are one‑hot encoded; numeric columns are passed through
    unchanged.  If scaling is desired, a StandardScaler could be added, but
    tree‑based models (RandomForest, XGB) generally do not require scaling.
    """
    categorical_cols = [
        'Crop_Type', 'Irrigation_Type', 'Soil_Type', 'Season'
    ]
    numeric_cols = [
        'Farm_Area(acres)', 'Fertilizer_Used(tons)', 'Pesticide_Used(kg)',
        'Water_Usage(cubic meters)'
    ]

    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    numeric_transformer = 'passthrough'

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_cols),
            ('num', numeric_transformer, numeric_cols),
        ]
    )

    return preprocessor


def evaluate_model(model_name: str, pipeline: Pipeline, X_train, y_train, X_test, y_test) -> dict:
    """Fit the pipeline, evaluate it using cross‑validation and on the test set.

    Returns a dictionary with model name and performance metrics (cross-val
    RMSE and test RMSE/R2).
    """
    # Cross‑validation on training set
    cv_scores = cross_val_score(
        pipeline, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error'
    )
    cv_rmse = -np.mean(cv_scores)

    # Fit on training data
    pipeline.fit(X_train, y_train)

    # Evaluate on test data
    y_pred = pipeline.predict(X_test)
    test_rmse = mean_squared_error(y_test, y_pred, squared=False)
    test_r2 = r2_score(y_test, y_pred)

    return {
        'model': model_name,
        'cv_rmse': cv_rmse,
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'pipeline': pipeline
    }


def train_models(df: pd.DataFrame):
    """Train multiple regression models and select the best one."""
    X = df.drop(columns=['Yield(tons)', 'Farm_ID'])
    y = df['Yield(tons)']

    # Build preprocessor
    preprocessor = build_preprocessor(df)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define models to evaluate
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=200, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(random_state=42)
    }
    if XGBRegressor is not None:
        models['XGBRegressor'] = XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, random_state=42
        )

    results = []
    best_score = float('inf')
    best_result = None

    for name, model in models.items():
        # Create a pipeline that applies the preprocessor then the model
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        res = evaluate_model(name, pipeline, X_train, y_train, X_test, y_test)
        results.append(res)
        print(f"{name}: CV RMSE={res['cv_rmse']:.2f}, Test RMSE={res['test_rmse']:.2f}, R2={res['test_r2']:.2f}")
        if res['cv_rmse'] < best_score:
            best_score = res['cv_rmse']
            best_result = res

    # Save preprocessor and best model pipeline
    joblib.dump(best_result['pipeline'].named_steps['preprocessor'], PREPROCESSOR_PATH)
    joblib.dump(best_result['pipeline'].named_steps['model'], MODEL_PATH)

    # Save results to CSV
    pd.DataFrame(results).drop(columns=['pipeline']).to_csv(RESULTS_PATH, index=False)

    print(f"\nBest model: {best_result['model']}")
    print(f"Saved preprocessor to {PREPROCESSOR_PATH} and model to {MODEL_PATH}")


# Optional: LSTM example

def build_and_train_lstm(X, y, epochs=100, batch_size=16):
    """Example function to build and train an LSTM model.

    Note: This function requires TensorFlow/Keras. It is provided for
    illustration only and is not called by default.  To use it, ensure
    that TensorFlow is installed and uncomment the relevant lines.
    """
    from sklearn.preprocessing import MinMaxScaler
    import tensorflow as tf
    from tensorflow.keras import layers, models

    # Flatten categorical variables by one‑hot encoding and scaling numeric
    preprocessor = build_preprocessor(pd.concat([X, y], axis=1))
    X_transformed = preprocessor.fit_transform(X)

    # Scale features to [0,1] for LSTM
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_transformed)

    # Reshape for LSTM: samples x timesteps x features; use timesteps=1
    X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

    # Build LSTM model
    model = models.Sequential()
    model.add(layers.LSTM(64, activation='relu', input_shape=(X_scaled.shape[1], X_scaled.shape[2])))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # Train LSTM
    model.fit(X_scaled, y.values, epochs=epochs, batch_size=batch_size, verbose=1)

    return model, preprocessor, scaler


def main():
    df = load_data(DATA_PATH)
    train_models(df)


if __name__ == '__main__':
    main()
