"""
Model explainability using SHAP values.
"""

import pandas as pd
import numpy as np
import joblib
import shap
from pathlib import Path
from utils import get_models_dir, get_processed_data_dir, get_outputs_dir, ensure_dir_exists
import matplotlib.pyplot as plt
import seaborn as sns


def load_model_and_data():
    """Load trained model and modeling dataset."""
    models_dir = get_models_dir()
    processed_dir = get_processed_data_dir()
    
    # Load XGBoost model (best performing)
    model = joblib.load(models_dir / 'xgboost_model.pkl')
    
    # Load modeling dataset
    df = pd.read_csv(processed_dir / 'modeling_dataset.csv')
    
    return model, df


def prepare_shap_data(df):
    """Prepare data for SHAP analysis."""
    # Separate features
    target_col = 'future_6_month_revenue'
    id_col = 'customerid'
    feature_cols = [col for col in df.columns if col not in [target_col, id_col]]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Log-transform target
    y_log = np.log1p(y)
    
    return X, y_log, feature_cols


def get_feature_names(preprocessor, original_features):
    """Get feature names after preprocessing."""
    # Get numerical feature names
    numerical_cols = preprocessor.named_transformers_['num'].get_feature_names_out()
    
    # Get categorical feature names
    cat_encoder = preprocessor.named_transformers_['cat']
    cat_features = cat_encoder.get_feature_names_out()
    
    # Combine
    all_features = list(numerical_cols) + list(cat_features)
    
    return all_features


def generate_shap_explanations(model, X, feature_names):
    """
    Generate SHAP values for the model.
    
    Args:
        model: Trained model pipeline
        X (pd.DataFrame): Feature dataframe
        feature_names (list): List of feature names after preprocessing
        
    Returns:
        shap.Explanation: SHAP values
    """
    print("Generating SHAP values...")
    
    # Get preprocessed data
    X_processed = model.named_steps['preprocessor'].transform(X)
    
    # Get the actual model (XGBoost)
    xgb_model = model.named_steps['model']
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(xgb_model)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X_processed)
    
    # Create Explanation object
    explanation = shap.Explanation(
        values=shap_values,
        base_values=explainer.expected_value,
        data=X_processed,
        feature_names=feature_names
    )
    
    print("SHAP values generated successfully")
    
    return explanation


def plot_feature_importance(explanation, save_path):
    """Plot global feature importance."""
    plt.figure(figsize=(12, 8))
    shap.plots.bar(explanation, show=False)
    plt.title('Global Feature Importance (SHAP)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Feature importance plot saved to: {save_path}")


def plot_shap_summary(explanation, save_path):
    """Plot SHAP summary plot."""
    plt.figure(figsize=(12, 8))
    shap.summary_plot(explanation, show=False)
    plt.title('SHAP Summary Plot', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"SHAP summary plot saved to: {save_path}")


def plot_shap_beeswarm(explanation, save_path):
    """Plot SHAP beeswarm plot."""
    plt.figure(figsize=(12, 8))
    shap.plots.beeswarm(explanation, show=False)
    plt.title('SHAP Beeswarm Plot', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"SHAP beeswarm plot saved to: {save_path}")


def explain_customer_prediction(model, X, customer_id, feature_names, original_df):
    """
    Explain prediction for a specific customer.
    
    Args:
        model: Trained model
        X (pd.DataFrame): Features
        customer_id: Customer ID to explain
        feature_names (list): Feature names
        original_df (pd.DataFrame): Original dataframe with customer IDs
        
    Returns:
        dict: Explanation details
    """
    # Find customer index
    customer_idx = original_df[original_df['customerid'] == customer_id].index
    if len(customer_idx) == 0:
        print(f"Customer {customer_id} not found")
        return None
    
    customer_idx = customer_idx[0]
    
    # Get preprocessed data
    X_processed = model.named_steps['preprocessor'].transform(X)
    customer_data = X_processed[customer_idx:customer_idx+1]
    
    # Get prediction
    prediction_log = model.predict(X.iloc[[customer_idx]])[0]
    prediction = np.expm1(prediction_log)
    
    # Get SHAP values for this customer
    xgb_model = model.named_steps['model']
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(customer_data)[0]
    
    # Get feature importance for this customer
    feature_importance = dict(zip(feature_names, shap_values))
    
    # Sort by absolute importance
    sorted_importance = sorted(
        feature_importance.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    
    explanation = {
        'customer_id': customer_id,
        'predicted_revenue_log': prediction_log,
        'predicted_revenue': prediction,
        'top_features': sorted_importance[:10]
    }
    
    return explanation


def save_explanation_summary(explanations, save_path):
    """Save explanation summary to CSV."""
    summary_data = []
    
    for exp in explanations:
        for feature, importance in exp['top_features']:
            summary_data.append({
                'customer_id': exp['customer_id'],
                'predicted_revenue': exp['predicted_revenue'],
                'feature': feature,
                'shap_value': importance
            })
    
    df = pd.DataFrame(summary_data)
    df.to_csv(save_path, index=False)
    print(f"Explanation summary saved to: {save_path}")


def main():
    """Main explainability pipeline."""
    print("=" * 60)
    print("SHAP Explainability Analysis")
    print("=" * 60)
    
    # Load model and data
    model, df = load_model_and_data()
    
    # Prepare data
    X, y, feature_names_original = prepare_shap_data(df)
    
    # Get feature names after preprocessing
    preprocessor = model.named_steps['preprocessor']
    feature_names = get_feature_names(preprocessor, feature_names_original)
    
    print(f"\nFeatures after preprocessing: {len(feature_names)}")
    
    # Generate SHAP explanations
    explanation = generate_shap_explanations(model, X, feature_names)
    
    # Save plots
    outputs_dir = get_outputs_dir()
    figures_dir = outputs_dir / 'figures'
    ensure_dir_exists(figures_dir)
    
    plot_feature_importance(explanation, figures_dir / 'shap_feature_importance.png')
    plot_shap_summary(explanation, figures_dir / 'shap_summary.png')
    plot_shap_beeswarm(explanation, figures_dir / 'shap_beeswarm.png')
    
    # Explain sample customers
    print("\n" + "=" * 60)
    print("Customer-Level Explanations")
    print("=" * 60)
    
    # Select a few sample customers
    sample_customers = df['customerid'].head(5).tolist()
    explanations = []
    
    for customer_id in sample_customers:
        exp = explain_customer_prediction(model, X, customer_id, feature_names, df)
        if exp:
            explanations.append(exp)
            print(f"\nCustomer {customer_id}:")
            print(f"  Predicted Revenue: £{exp['predicted_revenue']:.2f}")
            print(f"  Top 5 Factors:")
            for feature, importance in exp['top_features'][:5]:
                direction = "increases" if importance > 0 else "decreases"
                print(f"    {feature}: {direction} prediction ({importance:.4f})")
    
    # Save explanation summary
    save_explanation_summary(explanations, figures_dir / 'customer_explanations.csv')
    
    # Save SHAP values for later use
    joblib.dump(explanation, outputs_dir / 'figures' / 'shap_explanation.pkl')
    print("\nSHAP explanation saved for dashboard use")
    
    print("\n" + "=" * 60)
    print("Explainability Analysis Complete")
    print("=" * 60)
    
    return explanation


if __name__ == "__main__":
    explanation = main()
