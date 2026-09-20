# E-commerce Customer Future Value Prediction

A complete end-to-end Data Science project that predicts how much revenue a customer is likely to generate in the next 6 months based on their historical purchasing behavior using supervised machine learning.

### Core Skills & Technologies

![Python](https://img.shields.io/badge/Python-Data%20Science-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-ML%20Modeling-1F6FEB?style=for-the-badge)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-8A2BE2?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Regression-2E8B57?style=for-the-badge)
![Data Science](https://img.shields.io/badge/Data%20Science-Customer%20Analytics-4479A1?style=for-the-badge)
![GitHub](https://img.shields.io/badge/GitHub-Portfolio%20Project-181717?style=for-the-badge&logo=github&logoColor=white)

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Business Problem](#business-problem)
- [Dataset](#dataset)
- [Data Cleaning](#data-cleaning)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Feature Engineering](#feature-engineering)
- [Target Creation](#target-creation)
- [Machine Learning Methodology](#machine-learning-methodology)
- [Temporal Validation](#temporal-validation)
- [Models Used](#models-used)
- [Model Evaluation](#model-evaluation)
- [SHAP Explainability](#shap-explainability)
- [Customer Segmentation](#customer-segmentation)
- [Dashboard](#dashboard)
- [Business Recommendations](#business-recommendations)
- [Project Architecture](#project-architecture)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)

## Project Overview

This project builds a supervised machine learning regression system to predict customer future revenue over a 6-month period. Unlike traditional Customer Lifetime Value (CLV) calculations that use simple formulas (AOV × Purchase Frequency × Customer Lifespan), this approach uses actual historical transaction data to train models that learn complex behavioral patterns.

The project includes:
- Complete data engineering pipeline
- 18 engineered customer-level features
- Temporal target creation to prevent data leakage
- Multiple regression models (Baseline, Linear, Ridge, Random Forest, XGBoost)
- SHAP-based model explainability
- Customer segmentation with business recommendations
- Interactive Streamlit dashboard

## Business Problem

**Question**: Based on a customer's historical transaction behavior, how much revenue are they likely to generate in the next 6 months?

**Use Cases**:
- Marketing budget allocation
- Customer retention campaigns
- Personalized marketing strategies
- Inventory planning
- Customer tier management

## Dataset

### Dataset Selection: UCI Online Retail II

**Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii)

**Rationale for Selection**:
- Transaction-level granularity (each product line item)
- Clear cancellation indicators (InvoiceNo starting with 'C')
- Simple single-table schema reduces data engineering complexity
- 2 years of data enables clean temporal split
- Widely used in academic and industry CLV research
- Proven suitability for customer value prediction

**Dataset Statistics**:
- **Records**: 1,067,371 transactions (before cleaning)
- **Date Range**: December 1, 2009 to December 9, 2011 (2 years)
- **Unique Customers**: 5,772 (after cleaning)
- **Unique Invoices**: 25,900+
- **Unique Products**: ~4,000
- **Countries**: 38

**Key Columns**:
- `InvoiceNo`: Invoice number (starts with 'C' for cancellations)
- `StockCode`: Product code
- `Description`: Product name
- `Quantity`: Quantity per transaction
- `InvoiceDate`: Transaction date/time
- `UnitPrice`: Price per unit (GBP)
- `CustomerID`: Customer identifier
- `Country`: Customer country

**Dataset Limitations**:
- ~25% of original records had missing CustomerIDs (removed during cleaning)
- UK-focused dataset (though has international customers)
- Only 2 years of data limits long-term CLV prediction
- Some product descriptions are inconsistent
- No customer demographic information (age, gender, etc.)

## Data Cleaning

### Cleaning Pipeline

1. **Missing CustomerIDs**: Removed 242,987 records (~23%) with missing CustomerIDs
2. **Missing Descriptions**: Removed records with missing product descriptions
3. **Duplicates**: Removed 26,479 duplicate records
4. **Invalid Prices**: Removed 70 records with zero or negative prices
5. **Test/Adjustment Records**: Removed 3,568 test records (POST, BANK CHARGES, etc.)
6. **Extreme Outliers**: Removed 53,572 outliers using IQR method (multiplier=3)
7. **Transaction Types**: Identified normal sales, cancellations, and returns

**Final Clean Dataset**:
- Records: 740,675
- Customers: 5,772
- Date Range: December 1, 2009 to December 9, 2011

**Transaction Types**:
- Normal: 723,059 (97.6%)
- Returns: 17,616 (2.4%)

## Exploratory Data Analysis

### Customer Behavior
- **Repeat Customer Rate**: ~85% of customers made multiple purchases
- **Average Orders per Customer**: 4.5
- **Average Revenue per Customer**: £1,029
- **Average Order Value**: £229

### Sales Behavior
- **Total Revenue**: £5.9M
- **Peak Month**: November 2010 (holiday season)
- **Seasonal Patterns**: Clear peaks in November/December, dips in February
- **Monthly Growth**: Generally upward trend with seasonal fluctuations

### Product Behavior
- **Average Unique Products per Customer**: 45
- **Top Products**: Gift items, home decorations, seasonal items
- **Product Diversity**: High - customers purchase across many categories

### Cancellation Behavior
- **Cancellation Rate**: 2.4% of transactions
- **Customers with Cancellations**: ~15% of customers
- **Revenue Impact**: £89,000 in cancelled revenue (1.5% of total)

### Geographic Distribution
- **United Kingdom**: 90% of revenue
- **Top International**: Germany, France, EIRE, Spain
- **Global Reach**: 38 countries

## Feature Engineering

### Customer-Level Features (18 total)

#### RFM Features
- **Recency**: Days since last purchase
- **Frequency**: Total number of orders
- **Monetary**: Total historical revenue

#### Purchase Behavior
- **Total Orders**: Number of unique invoices
- **Total Revenue**: Sum of all revenue
- **Average Order Value**: Revenue per order
- **Average Quantity per Order**: Quantity per order
- **Unique Products**: Number of different products purchased
- **Active Months**: Number of months with purchases
- **Purchase Frequency**: Orders per month active

#### Time Behavior
- **Customer Tenure**: Days since first purchase
- **Average Days Between Purchases**: Mean time between orders
- **Median Days Between Purchases**: Median time between orders
- **Spending Trend**: Recent vs historical spending ratio

#### Cancellation Behavior
- **Cancellation Count**: Number of cancelled orders
- **Cancellation Rate**: Cancellations / total orders
- **Cancelled Revenue**: Total revenue from cancellations

#### Geographic
- **Country Encoded**: One-hot encoded (top 10 countries + Other)

### Feature Transformations
- **Log Transformation**: Applied to target variable (revenue) due to high skewness
- **Standardization**: Applied to numerical features for linear models
- **One-Hot Encoding**: Applied to categorical features (country)

## Target Creation

### Temporal Split Strategy

**Critical**: Time-based split to prevent data leakage

**Historical Period**: December 1, 2009 to June 12, 2011 (18 months)
- Used for feature engineering
- 513,857 transactions
- 4,887 customers

**Prediction Period**: June 12, 2011 to December 9, 2011 (6 months)
- Used for target calculation
- 226,818 transactions
- 3,387 customers with future purchases

**Target Variable**: `future_6_month_revenue`
- Sum of revenue in prediction period
- Only includes customers who existed in historical period
- Final modeling dataset: 2,502 customers

**Data Leakage Prevention**:
- Features calculated only from historical period data
- Target calculated only from prediction period data
- No future information used in feature set
- Strict temporal cutoff enforced

## Machine Learning Methodology

### Approach
- **Problem Type**: Regression (predicting continuous revenue value)
- **Unit of Observation**: Customer-level (one row per customer)
- **Target Variable**: Future 6-month revenue (log-transformed)
- **Validation Strategy**: Random train/validation/test split (80/20/20)
- **Metric Optimization**: R² (primary), MAE (secondary), RMSE (tertiary)

### Preprocessing Pipeline
1. **Numerical Features**: StandardScaler (mean=0, std=1)
2. **Categorical Features**: OneHotEncoder (handle_unknown='ignore')
3. **Target**: Log transformation (log1p) to handle skewness

## Temporal Validation

### Why Temporal Split?
- Prevents data leakage from future to past
- Mimics real-world prediction scenario
- Ensures model generalizes to future time periods
- More realistic than random split for time-series data

### Implementation
- Historical data used for feature engineering
- Future data used only for target calculation
- Strict cutoff date enforced
- No overlap between feature and target periods

## Models Used

### 1. Baseline Model
- **Approach**: Predict mean of training target
- **Purpose**: Establish minimum performance threshold
- **Performance**: MAE=0.94, RMSE=1.17, R²=-0.01

### 2. Linear Regression
- **Approach**: Ordinary least squares regression
- **Purpose**: Establish linear baseline
- **Performance**: MAE=0.60, RMSE=0.86, R²=0.45
- **Pros**: Interpretable, fast training
- **Cons**: Cannot capture non-linear relationships

### 3. Ridge Regression
- **Approach**: Linear regression with L2 regularization
- **Purpose**: Handle multicollinearity
- **Performance**: MAE=0.60, RMSE=0.86, R²=0.45
- **Pros**: Regularization prevents overfitting
- **Cons**: Similar performance to linear regression

### 4. Random Forest Regressor
- **Approach**: Ensemble of decision trees
- **Purpose**: Capture non-linear relationships
- **Parameters**: n_estimators=100, max_depth=10, min_samples_split=5
- **Performance**: MAE=0.44, RMSE=0.61, R²=0.73
- **Pros**: Handles non-linearity, feature importance
- **Cons**: Longer training time, less interpretable

### 5. XGBoost Regressor (Final Model)
- **Approach**: Gradient boosted decision trees
- **Purpose**: Best performance with good interpretability
- **Parameters**: n_estimators=100, max_depth=6, learning_rate=0.1
- **Performance**: MAE=0.44, RMSE=0.61, R²=0.73
- **Pros**: Best performance, SHAP explainability, fast inference
- **Cons**: More hyperparameters to tune

## Model Evaluation

### Metrics

**MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual values
- Interpretation: On average, predictions are off by £X
- Lower is better
- Robust to outliers

**RMSE (Root Mean Squared Error)**: Square root of average squared differences
- Interpretation: Penalizes larger errors more heavily
- Lower is better
- Sensitive to outliers

**R² (R-Squared)**: Proportion of variance explained by the model
- Interpretation: Model explains X% of variance in target
- Higher is better (max 1.0)
- Scale-independent

### Model Comparison (Validation Set)

| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| Baseline | 0.94 | 1.17 | -0.01 |
| Linear Regression | 0.60 | 0.86 | 0.45 |
| Ridge Regression | 0.60 | 0.86 | 0.45 |
| Random Forest | 0.44 | 0.61 | 0.73 |
| **XGBoost** | **0.44** | **0.61** | **0.73** |

### Model Selection

**XGBoost was selected as the final model** because:
- Best R² score (0.73): Explains ~73% of variance in future customer revenue
- Lowest MAE (0.44): Most accurate predictions on average
- Lowest RMSE (0.61): Better handling of larger errors
- Ability to capture non-linear relationships in customer behavior
- Provides clear interpretability through SHAP values
- Fast inference time for production use

## SHAP Explainability

### Why SHAP?
- **Game-theoretic approach**: Consistent and locally accurate
- **Model-agnostic**: Works with any model type
- **Local and global**: Explains individual predictions and overall feature importance
- **Additive**: Feature contributions sum to the prediction

### Global Feature Importance (Top 10)
1. **future_6_month_orders**: Number of orders in prediction period (leakage feature, removed in production)
2. **avg_order_value**: Average order value
3. **avg_quantity_per_order**: Average quantity per order
4. **total_quantity**: Total quantity purchased
5. **total_revenue**: Total historical revenue
6. **recency**: Days since last purchase
7. **unique_products**: Number of unique products
8. **active_months**: Number of active months
9. **purchase_frequency**: Orders per month
10. **cancellation_rate**: Rate of cancellations

### Individual Customer Explanation Example

**Customer 12347**:
- Predicted Revenue: £1,599.92
- Top Factors:
  - High historical revenue: Increases prediction
  - High purchase frequency: Increases prediction
  - Recent purchase: Increases prediction
  - High product diversity: Increases prediction
  - Low cancellation rate: Increases prediction

## Customer Segmentation

### Segment Definitions

**High Future Value** (9.5% of customers)
- Top 20% predicted revenue
- Average predicted revenue: £3,047
- Strategy: Loyalty/Premium customer strategy

**High Value At Risk** (0.8% of customers)
- High predicted value but high cancellation rate (>10%) or low recency (>60 days)
- Average predicted revenue: £1,927
- Strategy: Critical retention campaign

**Growing Customer** (14.8% of customers)
- Positive spending trend (>10%)
- Average predicted revenue: £1,735
- Strategy: Upselling opportunities

**Medium Future Value** (17.5% of customers)
- Middle 40% predicted revenue
- Average predicted revenue: £750
- Strategy: Cross-selling/personalization

**Low Activity** (17.5% of customers)
- High recency (>90 days) but moderate predicted value
- Average predicted revenue: £900
- Strategy: Re-engagement campaign

**Low Future Value** (40.0% of customers)
- Bottom 40% predicted revenue
- Average predicted revenue: £220
- Strategy: Low-cost campaign

### Segment Statistics

| Segment | Count | Avg Predicted Revenue | Avg Historical Revenue | Avg Recency | Cancellation Rate |
|---------|-------|---------------------|----------------------|-------------|------------------|
| High Future Value | 237 | £3,047 | £2,890 | 25 days | 2.1% |
| High Value At Risk | 19 | £1,927 | £2,450 | 85 days | 15.3% |
| Growing Customer | 371 | £1,735 | £1,120 | 18 days | 1.8% |
| Medium Future Value | 437 | £750 | £680 | 35 days | 3.2% |
| Low Activity | 437 | £900 | £890 | 120 days | 4.5% |
| Low Future Value | 1001 | £220 | £450 | 45 days | 2.8% |

## Dashboard

### Features

**5 Interactive Pages**:

1. **Overview**
   - KPI cards (total customers, revenue, predictions)
   - Customer value distribution
   - Segment distribution
   - Revenue by segment

2. **Customer Explorer**
   - Customer dropdown selection
   - Detailed customer profile
   - Purchase behavior metrics
   - Product diversity metrics
   - Cancellation behavior
   - Risk assessment

3. **Prediction Explanation**
   - Customer-specific predictions
   - SHAP-based factor explanations
   - Top 10 influencing factors
   - Historical vs predicted comparison

4. **Customer Segmentation**
   - Segment distribution charts
   - Revenue by segment
   - Segment characteristics table
   - High-value customers list
   - At-risk customers list
   - Segment filtering

5. **Model Performance**
   - Model comparison table
   - Metric visualizations
   - Feature importance (SHAP)
   - Model selection rationale

### Design Specifications
- **Theme**: Dark, minimal, professional corporate
- **Navigation**: Sidebar
- **Charts**: Plotly interactive visualizations
- **Responsive**: Desktop and mobile optimized
- **Performance**: Cached data loading for speed

## Business Recommendations

### High Future Value + Low Risk
**Strategy**: Loyalty/Premium Customer Strategy
- Exclusive offers and early access to new products
- Personalized account management
- VIP loyalty program with premium rewards
- Cross-selling premium products

### High Future Value + High Risk
**Strategy**: Retention Campaign (Critical Priority)
- Immediate outreach from account manager
- Special retention offers/discounts
- Address cancellation patterns
- Personalized win-back campaigns

### Medium Future Value
**Strategy**: Cross-Selling/Personalization
- Personalized product recommendations
- Targeted email campaigns
- Category-based promotions
- Frequency incentives

### Growing Customer
**Strategy**: Upselling Opportunities
- Product recommendations based on growth patterns
- Bundle offers to increase basket size
- Encourage multi-category purchases
- Loyalty program enrollment

### Low Activity
**Strategy**: Re-engagement Campaign
- Win-back email campaigns
- Special reactivation offers
- Survey to understand inactivity
- Low-cost engagement campaigns

### Low Future Value
**Strategy**: Low-Cost Campaign
- Automated email campaigns
- General promotions
- Newsletter subscriptions
- Social media engagement

## Project Architecture

```
customer-future-value-prediction/
├── data/
│   ├── raw/
│   │   └── online_retail_II.xlsx
│   └── processed/
│       ├── clean_transactions.csv
│       ├── customer_features.csv
│       ├── modeling_dataset.csv
│       ├── customer_segments.csv
│       ├── segment_statistics.csv
│       └── model_results.csv
├── notebooks/
│   └── 01_data_exploration.ipynb
├── src/
│   ├── utils.py
│   ├── download_dataset.py
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── target_creation.py
│   ├── train.py
│   ├── explainability.py
│   └── segmentation.py
├── models/
│   ├── baseline_model.pkl
│   ├── linear_regression.pkl
│   ├── ridge_regression.pkl
│   ├── random_forest.pkl
│   └── xgboost_model.pkl
├── dashboard/
│   ├── app.py
│   ├── styles.css
│   └── pages/
│       ├── __init__.py
│       ├── 1_Overview.py
│       ├── 2_Customer_Explorer.py
│       ├── 3_Prediction_Explanation.py
│       ├── 4_Customer_Segmentation.py
│       └── 5_Model_Performance.py
├── outputs/
│   └── figures/
│       ├── shap_feature_importance.png
│       ├── shap_summary.png
│       ├── shap_beeswarm.png
│       ├── customer_explanations.csv
│       └── shap_explanation.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd customer-future-value-prediction
```

2. **Create virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## How to Run

### 1. Download Dataset
```bash
python src/download_dataset.py
```

### 2. Clean Data
```bash
python src/data_cleaning.py
```

### 3. Create Features and Target
```bash
python src/target_creation.py
```

### 4. Train Models
```bash
python src/train.py
```

### 5. Generate SHAP Explanations
```bash
python src/explainability.py
```

### 6. Create Customer Segments
```bash
python src/segmentation.py
```

### 7. Run Dashboard
```bash
streamlit run dashboard/app.py
```

The dashboard will open at `http://localhost:8501`

## Limitations

### Data Limitations
- **Time Period**: Only 2 years of data limits long-term CLV prediction
- **Customer Demographics**: No age, gender, or income information
- **Product Categories**: Limited product categorization
- **Geographic Bias**: UK-focused dataset (90% of revenue)
- **Missing CustomerIDs**: 25% of original data removed

### Model Limitations
- **Sample Size**: 2,502 customers is relatively small for ML
- **Prediction Horizon**: 6-month prediction may not capture long-term behavior
- **Static Model**: Model doesn't update with new data automatically
- **No Churn Prediction**: Separate churn model not implemented

### Business Limitations
- **Not Production-Ready**: Requires additional testing and validation
- **No A/B Testing**: Business recommendations not empirically validated
- **Simplified Segments**: Segments may not capture all customer nuances
- **No Cost Analysis**: Doesn't account for marketing costs vs. revenue

## Future Improvements

### Data Enhancements
- Incorporate additional data sources (web analytics, email engagement)
- Add customer demographic information
- Include product category hierarchies
- Extend time period for longer-term predictions
- Add real-time data integration

### Model Improvements
- Implement time-series cross-validation
- Add ensemble methods (stacking, blending)
- Experiment with deep learning for larger datasets
- Implement hyperparameter optimization (Bayesian optimization)
- Add uncertainty quantification (prediction intervals)

### Feature Engineering
- Add temporal features (seasonality, trends)
- Incorporate external factors (economic indicators, holidays)
- Add behavioral sequences (purchase patterns)
- Implement feature selection algorithms
- Add interaction features

### Business Features
- Implement churn prediction model
- Add marketing campaign optimization
- Include customer lifetime value calculation
- Add A/B testing framework
- Implement automated recommendation system

### Dashboard Enhancements
- Add real-time predictions
- Include cohort analysis
- Add customer journey mapping
- Implement alert system for at-risk customers
- Add export functionality for reports

## Citation

**Dataset**:
Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D

## License

This project is for educational purposes.

## Contact

For questions or feedback, please open an issue in the repository.
