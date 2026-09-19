"""
Train and evaluate machine learning models for customer future value prediction.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib
from utils import get_processed_data_dir, get_models_dir, ensure_dir_exists


def load_modeling_dataset():
    """Load the modeling dataset."""
    processed_dir = get_processed_data_dir()
    df = pd.read_csv(processed_dir / 'modeling_dataset.csv')
    print(f"Loaded modeling dataset: {df.shape}")
    return df


def prepare_features(df):
    """
    Prepare features for modeling.
    
    Args:
        df (pd.DataFrame): Modeling dataset
        
    Returns:
        tuple: (X, y, feature_names)
    """
    # Separate features and target
    target_col = 'future_6_month_revenue'
    id_col = 'customerid'
    
    feature_cols = [col for col in df.columns if col not in [target_col, id_col]]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Log-transform target (revenue is highly skewed)
    y_log = np.log1p(y)
    
    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Target statistics:")
    print(f"  Original - Mean: {y.mean():.2f}, Median: {y.median():.2f}")
    print(f"  Log-transformed - Mean: {y_log.mean():.2f}, Median: {y_log.median():.2f}")
    
    return X, y, y_log, feature_cols


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into train/validation/test sets using time-based split.
    Since we already did temporal split in target creation, we'll do random split here.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        test_size (float): Proportion for test set
        random_state (int): Random seed
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Second split: train vs val
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=test_size, random_state=random_state
    )
    
    print(f"\nData split:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_preprocessor(X):
    """
    Create preprocessing pipeline for features.
    
    Args:
        X (pd.DataFrame): Feature dataframe
        
    Returns:
        ColumnTransformer: Preprocessor
    """
    # Identify numerical and categorical columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    print(f"\nFeature types:")
    print(f"  Numerical: {len(numerical_cols)}")
    print(f"  Categorical: {len(categorical_cols)}")
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ],
        remainder='passthrough'
    )
    
    return preprocessor


def train_baseline_model(y_train):
    """
    Train baseline model (predict mean).
    
    Args:
        y_train (pd.Series): Training target
        
    Returns:
        dict: Baseline model info
    """
    mean_value = y_train.mean()
    
    baseline = {
        'type': 'baseline',
        'mean_value': mean_value
    }
    
    return baseline


def predict_baseline(baseline, X):
    """Predict using baseline model."""
    return np.full(len(X), baseline['mean_value'])


def train_linear_regression(X_train, y_train, preprocessor):
    """Train Linear Regression model."""
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LinearRegression())
    ])
    
    pipeline.fit(X_train, y_train)
    return pipeline


def train_ridge_regression(X_train, y_train, preprocessor):
    """Train Ridge Regression model."""
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', Ridge(alpha=1.0))
    ])
    
    pipeline.fit(X_train, y_train)
    return pipeline


def train_random_forest(X_train, y_train, preprocessor):
    """Train Random Forest Regressor."""
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    pipeline.fit(X_train, y_train)
    return pipeline


def train_xgboost(X_train, y_train, preprocessor):
    """Train XGBoost Regressor."""
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(model, X, y, model_name, is_baseline=False):
    """
    Evaluate model performance.
    
    Args:
        model: Trained model
        X (pd.DataFrame): Features
        y (pd.Series): True target values
        model_name (str): Model name
        is_baseline (bool): Whether this is baseline model
        
    Returns:
        dict: Evaluation metrics
    """
    if is_baseline:
        y_pred = predict_baseline(model, X)
    else:
        y_pred = model.predict(X)
    
    # Calculate metrics
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    
    metrics = {
        'model': model_name,
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }
    
    print(f"\n{model_name} Results:")
    print(f"  MAE: {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²: {r2:.4f}")
    
    return metrics


def save_model(model, model_name):
    """Save trained model to disk."""
    models_dir = get_models_dir()
    ensure_dir_exists(models_dir)
    
    model_path = models_dir / f"{model_name}.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("Training Machine Learning Models")
    print("=" * 60)
    
    # Load data
    df = load_modeling_dataset()
    
    # Prepare features
    X, y, y_log, feature_names = prepare_features(df)
    
    # Use log-transformed target for modeling
    y_target = y_log
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y_target)
    
    # Create preprocessor
    preprocessor = create_preprocessor(X)
    
    # Train baseline model
    print("\n" + "=" * 60)
    print("Training Baseline Model")
    print("=" * 60)
    baseline = train_baseline_model(y_train)
    
    # Train Linear Regression
    print("\n" + "=" * 60)
    print("Training Linear Regression")
    print("=" * 60)
    lr_model = train_linear_regression(X_train, y_train, preprocessor)
    
    # Train Ridge Regression
    print("\n" + "=" * 60)
    print("Training Ridge Regression")
    print("=" * 60)
    ridge_model = train_ridge_regression(X_train, y_train, preprocessor)
    
    # Train Random Forest
    print("\n" + "=" * 60)
    print("Training Random Forest")
    print("=" * 60)
    rf_model = train_random_forest(X_train, y_train, preprocessor)
    
    # Train XGBoost
    print("\n" + "=" * 60)
    print("Training XGBoost")
    print("=" * 60)
    xgb_model = train_xgboost(X_train, y_train, preprocessor)
    
    # Evaluate all models on validation set
    print("\n" + "=" * 60)
    print("Model Evaluation on Validation Set")
    print("=" * 60)
    
    results = []
    results.append(evaluate_model(baseline, X_val, y_val, "Baseline", is_baseline=True))
    results.append(evaluate_model(lr_model, X_val, y_val, "Linear Regression"))
    results.append(evaluate_model(ridge_model, X_val, y_val, "Ridge Regression"))
    results.append(evaluate_model(rf_model, X_val, y_val, "Random Forest"))
    results.append(evaluate_model(xgb_model, X_val, y_val, "XGBoost"))
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("Model Comparison (Validation Set)")
    print("=" * 60)
    print(results_df.to_string(index=False))
    
    # Save models
    print("\n" + "=" * 60)
    print("Saving Models")
    print("=" * 60)
    save_model(baseline, "baseline_model")
    save_model(lr_model, "linear_regression")
    save_model(ridge_model, "ridge_regression")
    save_model(rf_model, "random_forest")
    save_model(xgb_model, "xgboost_model")
    
    # Save results
    processed_dir = get_processed_data_dir()
    results_df.to_csv(processed_dir / 'model_results.csv', index=False)
    print(f"\nResults saved to: {processed_dir / 'model_results.csv'}")
    
    return results_df, lr_model, ridge_model, rf_model, xgb_model


if __name__ == "__main__":
    results, lr, ridge, rf, xgb = main()
