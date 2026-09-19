"""
Customer segmentation based on predicted future value and behavioral features.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from utils import get_processed_data_dir, get_models_dir, ensure_dir_exists


def load_model_and_predictions():
    """Load trained model and generate predictions."""
    models_dir = get_models_dir()
    processed_dir = get_processed_data_dir()
    
    # Load XGBoost model
    model = joblib.load(models_dir / 'xgboost_model.pkl')
    
    # Load modeling dataset
    df = pd.read_csv(processed_dir / 'modeling_dataset.csv')
    
    # Prepare features
    target_col = 'future_6_month_revenue'
    id_col = 'customerid'
    feature_cols = [col for col in df.columns if col not in [target_col, id_col]]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Generate predictions (log scale)
    y_pred_log = model.predict(X)
    y_pred = np.expm1(y_pred_log)
    
    # Add predictions to dataframe
    df['predicted_future_revenue'] = y_pred
    
    return df, model


def create_customer_segments(df):
    """
    Create customer segments based on predicted revenue and behavior.
    
    Segment Definitions:
    - High Future Value: Top 20% predicted revenue
    - Medium Future Value: Middle 40% predicted revenue
    - Low Future Value: Bottom 40% predicted revenue
    - High Value At Risk: High predicted value but high cancellation rate or low recency
    - Growing Customer: Positive spending trend
    - Low Activity: High recency (>90 days) but moderate predicted value
    """
    print("\n=== Creating Customer Segments ===")
    
    # Calculate quantiles for predicted revenue
    revenue_80 = df['predicted_future_revenue'].quantile(0.8)
    revenue_40 = df['predicted_future_revenue'].quantile(0.4)
    
    print(f"Revenue quantiles:")
    print(f"  80th percentile: £{revenue_80:.2f}")
    print(f"  40th percentile: £{revenue_40:.2f}")
    
    # Initialize segment column
    df['segment'] = 'Medium Future Value'
    
    # High Future Value (top 20%)
    df.loc[df['predicted_future_revenue'] >= revenue_80, 'segment'] = 'High Future Value'
    
    # Low Future Value (bottom 40%)
    df.loc[df['predicted_future_revenue'] < revenue_40, 'segment'] = 'Low Future Value'
    
    # High Value At Risk: High predicted value but high cancellation rate (>10%) or low recency (>60 days)
    high_value_mask = df['predicted_future_revenue'] >= revenue_80
    high_risk_mask = (df['cancellation_rate'] > 0.1) | (df['recency'] > 60)
    df.loc[high_value_mask & high_risk_mask, 'segment'] = 'High Value At Risk'
    
    # Growing Customer: Positive spending trend
    growing_mask = (df['spending_trend'] > 0.1) & (df['predicted_future_revenue'] >= revenue_40)
    df.loc[growing_mask, 'segment'] = 'Growing Customer'
    
    # Low Activity: High recency (>90 days)
    low_activity_mask = (df['recency'] > 90) & (df['predicted_future_revenue'] >= revenue_40)
    df.loc[low_activity_mask, 'segment'] = 'Low Activity'
    
    # Print segment distribution
    segment_counts = df['segment'].value_counts()
    print("\nSegment Distribution:")
    for segment, count in segment_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {segment}: {count} ({percentage:.1f}%)")
    
    # Calculate segment statistics
    segment_stats = df.groupby('segment').agg({
        'predicted_future_revenue': ['mean', 'median', 'sum'],
        'total_revenue': 'mean',
        'recency': 'mean',
        'cancellation_rate': 'mean',
        'customerid': 'count'
    }).round(2)
    
    segment_stats.columns = ['Avg Predicted Revenue', 'Median Predicted Revenue', 
                           'Total Predicted Revenue', 'Avg Historical Revenue',
                           'Avg Recency', 'Avg Cancellation Rate', 'Count']
    
    print("\nSegment Statistics:")
    print(segment_stats)
    
    return df, segment_stats


def save_segmented_data(df, segment_stats):
    """Save segmented data and statistics."""
    processed_dir = get_processed_data_dir()
    
    # Save segmented dataset
    df.to_csv(processed_dir / 'customer_segments.csv', index=False)
    print(f"\nSegmented data saved to: {processed_dir / 'customer_segments.csv'}")
    
    # Save segment statistics
    segment_stats.to_csv(processed_dir / 'segment_statistics.csv')
    print(f"Segment statistics saved to: {processed_dir / 'segment_statistics.csv'}")
    
    return processed_dir / 'customer_segments.csv'


def get_segment_recommendations():
    """
    Get business recommendations for each segment.
    
    Returns:
        dict: Recommendations per segment
    """
    recommendations = {
        'High Future Value': {
            'strategy': 'Loyalty/Premium Customer Strategy',
            'actions': [
                'Exclusive offers and early access to new products',
                'Personalized account management',
                'VIP loyalty program with premium rewards',
                'Cross-selling premium products'
            ],
            'priority': 'High'
        },
        'High Value At Risk': {
            'strategy': 'Retention Campaign',
            'actions': [
                'Immediate outreach from account manager',
                'Special retention offers/discounts',
                'Address cancellation patterns',
                'Personalized win-back campaigns'
            ],
            'priority': 'Critical'
        },
        'Growing Customer': {
            'strategy': 'Upselling Opportunities',
            'actions': [
                'Product recommendations based on growth patterns',
                'Bundle offers to increase basket size',
                'Encourage multi-category purchases',
                'Loyalty program enrollment'
            ],
            'priority': 'Medium-High'
        },
        'Medium Future Value': {
            'strategy': 'Cross-Selling/Personalization',
            'actions': [
                'Personalized product recommendations',
                'Targeted email campaigns',
                'Category-based promotions',
                'Frequency incentives'
            ],
            'priority': 'Medium'
        },
        'Low Activity': {
            'strategy': 'Re-engagement Campaign',
            'actions': [
                'Win-back email campaigns',
                'Special reactivation offers',
                'Survey to understand inactivity',
                'Low-cost engagement campaigns'
            ],
            'priority': 'Medium'
        },
        'Low Future Value': {
            'strategy': 'Low-Cost Campaign',
            'actions': [
                'Automated email campaigns',
                'General promotions',
                'Newsletter subscriptions',
                'Social media engagement'
            ],
            'priority': 'Low'
        }
    }
    
    return recommendations


def main():
    """Main segmentation pipeline."""
    print("=" * 60)
    print("Customer Segmentation")
    print("=" * 60)
    
    # Load model and generate predictions
    df, model = load_model_and_predictions()
    
    # Create segments
    df, segment_stats = create_customer_segments(df)
    
    # Save results
    save_segmented_data(df, segment_stats)
    
    # Get recommendations
    recommendations = get_segment_recommendations()
    
    print("\n" + "=" * 60)
    print("Business Recommendations by Segment")
    print("=" * 60)
    
    for segment, rec in recommendations.items():
        print(f"\n{segment}:")
        print(f"  Strategy: {rec['strategy']}")
        print(f"  Priority: {rec['priority']}")
        print(f"  Actions:")
        for action in rec['actions']:
            print(f"    - {action}")
    
    return df, segment_stats, recommendations


if __name__ == "__main__":
    df, stats, recs = main()
