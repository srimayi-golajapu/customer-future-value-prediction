# Interview Preparation - Customer Future Value Prediction Project

This document contains interview questions and answers for the Customer Future Value Prediction project.

## Table of Contents

- [30-45 Second Project Explanation](#30-45-second-project-explanation)
- [2-3 Minute Detailed Explanation](#2-3-minute-detailed-explanation)
- [Basic Questions](#basic-questions)
- [Data Questions](#data-questions)
- [ML Questions](#ml-questions)
- [Feature Engineering Questions](#feature-engineering-questions)
- [Explainability Questions](#explainability-questions)
- [Business Questions](#business-questions)

---

## 30-45 Second Project Explanation

"I built a machine learning system to predict customer future revenue over the next 6 months using historical transaction data. I used the UCI Online Retail II dataset with 740K transactions from 5,772 customers. I engineered 18 customer-level features including RFM metrics, purchase behavior, and cancellation patterns. I trained multiple regression models and selected XGBoost as the final model, achieving an R² of 0.73. I added SHAP explainability to understand feature importance, created customer segments with business recommendations, and built an interactive Streamlit dashboard for stakeholders."

---

## 2-3 Minute Detailed Explanation

"I built an end-to-end customer future value prediction system to help businesses forecast which customers will generate the most revenue in the next 6 months. This is critical for marketing budget allocation and customer retention strategies.

I used the UCI Online Retail II dataset, which contains 740K cleaned transactions from 5,772 customers over a 2-year period. I implemented a strict temporal split—using 18 months of historical data for feature engineering and the final 6 months for the target variable—to prevent data leakage.

I engineered 18 customer-level features including RFM metrics (recency, frequency, monetary), purchase behavior (average order value, product diversity), time behavior (customer tenure, spending trends), and cancellation behavior. The target variable is the sum of revenue in the 6-month prediction period.

I trained and compared five models: baseline, linear regression, ridge regression, random forest, and XGBoost. XGBoost performed best with an R² of 0.73, meaning it explains 73% of the variance in future customer revenue. I used SHAP values for model explainability, which showed that average order value, total revenue, and recency are the most important features.

I created six customer segments based on predicted revenue and behavioral patterns: High Future Value, High Value At Risk, Growing Customer, Medium Future Value, Low Activity, and Low Future Value. Each segment has specific business recommendations.

Finally, I built a 5-page interactive Streamlit dashboard with a dark theme that allows stakeholders to explore customer predictions, understand model explanations, view segment distributions, and analyze model performance."

---

## Basic Questions

### Why did you choose this project?

"I chose this project because customer lifetime value prediction is a fundamental business problem that demonstrates real-world impact. It combines data engineering, feature engineering, machine learning, and business intelligence—all critical skills for a data scientist. Unlike simple classification problems, this required handling temporal data, preventing data leakage, and creating interpretable models that business stakeholders can actually use."

### What is CLV?

"Customer Lifetime Value (CLV) represents the total revenue a business can expect from a single customer account throughout their relationship. Traditional CLV calculations use simple formulas like AOV × Purchase Frequency × Customer Lifespan, but this approach doesn't capture complex behavioral patterns. My project uses supervised machine learning to predict future revenue based on actual historical behavior, which is more accurate and nuanced."

### Why predict future value instead of calculating historical CLV?

"Predicting future value is more actionable for businesses than calculating historical CLV. Historical CLV tells you what a customer was worth, but future value tells you what they're likely to spend, which is what matters for budgeting, marketing campaigns, and retention strategies. Also, historical CLV formulas assume linear relationships and don't capture complex behavioral patterns that machine learning can learn."

### What is the target variable?

"The target variable is `future_6_month_revenue`, which is the sum of revenue a customer generates in the 6-month prediction period. I log-transformed this variable because revenue is highly skewed—most customers spend relatively little, but a few spend a lot. Log transformation makes the distribution more normal and helps models perform better."

### What is the unit of observation?

"The unit of observation is the customer. Each row in my modeling dataset represents one customer with their aggregated features and their future 6-month revenue target. I aggregated transaction-level data to customer-level because we're predicting customer-level outcomes, not transaction-level outcomes."

### Why did you choose this dataset?

"I chose the UCI Online Retail II dataset because it has transaction-level granularity, clear cancellation indicators, and a simple schema that allowed me to focus on the ML pipeline rather than data engineering. The 2-year time period enabled a clean temporal split for historical and prediction periods. It's also a well-known dataset in the ML community, making it easy to validate my approach against existing research."

---

## Data Questions

### How did you clean the data?

"I removed records with missing CustomerIDs (23% of data), missing descriptions, and duplicates. I filtered out invalid prices (zero or negative) and test/adjustment records like bank charges. I removed extreme outliers using the IQR method with a multiplier of 3. I also identified transaction types—normal sales, cancellations (InvoiceNo starting with 'C'), and returns (negative quantities). The final clean dataset had 740K records from 5,772 customers."

### How did you handle missing CustomerIDs?

"I removed all records with missing CustomerIDs because you can't build customer-level features without a customer identifier. This removed about 23% of the original data, which is significant but necessary. In a production setting, I might treat these as anonymous customers and build a separate model for them, but for this project, I focused on identified customers."

### How did you handle cancellations?

"I identified cancellations by checking if the InvoiceNo starts with 'C'. I created a separate `transaction_type` column to distinguish normal sales, cancellations, and returns. For feature engineering, I calculated cancellation-specific features like cancellation count, cancellation rate, and total cancelled revenue. These features help the model understand which customers have high cancellation patterns, which is predictive of future behavior."

### How did you handle outliers?

"I used the IQR (Interquartile Range) method with a multiplier of 3 to remove extreme outliers in quantity and unit price. This removed about 53K records. I only applied this to normal transactions, not cancellations or returns, since those naturally have negative values. This approach balances removing data quality issues while preserving legitimate high-value transactions."

### How did you prevent data leakage?

"I used a strict temporal split. I used data from December 2009 to June 2011 for feature engineering (historical period) and data from June 2011 to December 2011 for the target variable (prediction period). This ensures that no future information leaks into the features. I also made sure that only customers who existed in the historical period were included in the modeling dataset."

---

## ML Questions

### Why is this regression?

"This is a regression problem because we're predicting a continuous value—future revenue. Classification would be inappropriate because revenue can take any positive value, and we need to predict the actual amount, not just a category. Regression allows us to predict the exact revenue amount, which is what businesses need for budgeting and planning."

### Why Linear Regression?

"I included Linear Regression as a baseline to establish a minimum performance threshold. It's simple, interpretable, and fast to train. If a complex model doesn't significantly outperform linear regression, it suggests the problem might not require complex modeling. In this case, linear regression achieved R²=0.45, which showed there were non-linear patterns that more complex models could capture."

### Why Random Forest?

"I used Random Forest to capture non-linear relationships and interactions between features. It's an ensemble method that builds multiple decision trees and averages their predictions, which reduces overfitting compared to single trees. Random Forest also provides feature importance, which helps with interpretability. It achieved R²=0.73, significantly better than linear regression."

### Why XGBoost?

"I chose XGBoost as the final model because it achieved the best performance (R²=0.73) with good interpretability through SHAP values. XGBoost is a gradient boosting algorithm that builds trees sequentially, each correcting the errors of the previous one. It's highly efficient, handles missing values well, and provides excellent performance on tabular data. It also has built-in regularization to prevent overfitting."

### How did you split the data?

"I used an 80/20 train/validation/test split. First, I split 80% for training and 20% for testing. Then, I split the training set again into 80% training and 20% validation. This gave me three sets: training (64%), validation (16%), and test (20%). I used the validation set for hyperparameter tuning and model selection, and the test set for final evaluation."

### Why not randomly split transactions?

"Randomly splitting transactions would cause data leakage because transactions from the same customer would appear in both training and test sets. The model would learn customer-specific patterns that wouldn't generalize to new customers. By splitting at the customer level and using temporal splits, I ensure the model learns generalizable patterns, not customer-specific noise."

### What does MAE mean?

"MAE (Mean Absolute Error) is the average absolute difference between predicted and actual values. For this project, an MAE of 0.44 (on log-transformed revenue) means that on average, the model's predictions are off by about 0.44 in log space. When converted back to pounds, this represents a meaningful error in revenue prediction. MAE is robust to outliers and easy to interpret."

### What does RMSE mean?

"RMSE (Root Mean Squared Error) is the square root of the average squared differences between predicted and actual values. RMSE penalizes larger errors more heavily than MAE because of the squaring. An RMSE of 0.61 means the model has some larger errors, but overall performance is good. RMSE is useful when large errors are particularly undesirable."

### What does R² mean?

"R² (R-Squared) is the proportion of variance in the target variable that the model explains. An R² of 0.73 means the model explains 73% of the variance in future customer revenue. R² ranges from 0 to 1 (or can be negative for very poor models). It's a scale-independent metric that's useful for comparing models, though it doesn't tell you about prediction error in absolute terms."

### How did you tune the model?

"I used default hyperparameters for most models to establish baselines. For Random Forest, I used n_estimators=100, max_depth=10, and min_samples_split=5. For XGBoost, I used n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8, and colsample_bytree=0.8. In a production setting, I would use grid search or Bayesian optimization for more thorough hyperparameter tuning."

---

## Feature Engineering Questions

### Why did you use RFM?

"RFM (Recency, Frequency, Monetary) is a classic customer segmentation framework that's proven to be predictive of customer behavior. Recency measures how recently a customer purchased, frequency measures how often they purchase, and monetary measures how much they spend. These three dimensions capture the most important aspects of customer behavior and are standard in marketing analytics."

### Which features were most important?

"According to SHAP values, the most important features were average order value, total revenue, recency, unique products purchased, and purchase frequency. This makes sense—customers who spend more per order, have spent more historically, purchased recently, buy diverse products, and purchase frequently are likely to spend more in the future."

### How did you calculate recency?

"Recency is the number of days since the customer's last purchase relative to the reference date (the end of the historical period). I calculated it as: reference_date - last_purchase_date. Lower recency means more recent purchases, which is generally associated with higher future spending."

### How did you calculate frequency?

"I calculated frequency in two ways: total orders (count of unique invoices) and purchase frequency (orders per month active). Total orders captures overall engagement, while purchase frequency normalizes by how long the customer has been active, giving a sense of purchase intensity."

### How did you calculate customer tenure?

"Customer tenure is the number of days from the customer's first purchase to the reference date. I calculated it as: reference_date - first_purchase_date + 1. Longer tenure generally indicates more established customer relationships, though it doesn't always correlate with higher spending."

### How did you calculate future revenue?

"Future revenue is the sum of all revenue in the 6-month prediction period (June 2011 to December 2011) for each customer. I only included normal transactions (not cancellations or returns) in this calculation. This is the target variable that the model learns to predict."

---

## Explainability Questions

### Why SHAP?

"I chose SHAP (SHapley Additive exPlanations) because it provides consistent, locally accurate explanations for any machine learning model. SHAP values have a solid theoretical foundation in game theory and are additive—feature contributions sum to the model's prediction. This makes it easy to explain both individual predictions and global feature importance to stakeholders."

### How do SHAP values work?

"SHAP values calculate the contribution of each feature to a specific prediction by comparing the model's prediction with and without that feature. It does this by considering all possible feature combinations and assigning each feature a value that represents its marginal contribution. Positive SHAP values increase the prediction, negative values decrease it. The magnitude indicates the importance."

### How would you explain an individual prediction?

"For an individual customer, I would show their predicted revenue and the top factors influencing that prediction. For example: 'Customer 12347 is predicted to spend £1,600 in the next 6 months. The main factors driving this prediction are: high average order value (+£500), high total historical revenue (+£300), recent purchase (+£200), and low cancellation rate (+£100).' This gives stakeholders a clear, actionable explanation."

---

## Business Questions

### How can a company use the predictions?

"Companies can use these predictions for several purposes: 1) Marketing budget allocation—spend more on high-value customers, 2) Customer retention—identify at-risk high-value customers for retention campaigns, 3) Personalization—tailor marketing based on predicted value, 4) Inventory planning—anticipate demand from high-value segments, 5) Customer tier management—create VIP programs for high-value customers."

### What does high-value at-risk mean?

"High-value at-risk customers are those with high predicted future revenue but indicators of risk—either high cancellation rates (>10% of orders) or low recency (>60 days since last purchase). These customers are valuable to the business but may churn, so they require immediate retention efforts like personalized outreach or special offers."

### How would marketing use this model?

"Marketing teams could use this model to segment customers and design targeted campaigns. For high-value customers, they'd run loyalty programs and exclusive offers. For at-risk customers, they'd run retention campaigns. For growing customers, they'd run upselling campaigns. For low-value customers, they'd run low-cost automated campaigns. This ensures marketing spend is optimized for ROI."

### What are the limitations?

"The main limitations are: 1) Only 2 years of data limits long-term predictions, 2) No customer demographic information (age, income), 3) UK-focused dataset may not generalize globally, 4) Model doesn't automatically update with new data, 5) Predictions are for 6 months only, not lifetime value, 6) Business recommendations aren't empirically validated through A/B testing. In production, these would need to be addressed."

---

## Additional Tips for Interviews

### Technical Depth
- Be prepared to discuss the math behind your metrics (MAE, RMSE, R²)
- Understand the difference between parametric and non-parametric models
- Be ready to explain gradient boosting intuitively
- Know the trade-offs between different model types

### Business Focus
- Always connect technical decisions to business impact
- Be prepared to discuss ROI of your predictions
- Understand how your model fits into a larger business strategy
- Be ready to discuss implementation challenges

### Data Science Process
- Emphasize the importance of data quality
- Discuss how you'd handle production data drift
- Be prepared to discuss monitoring and retraining strategies
- Understand the ethical implications of customer segmentation

### Communication
- Practice explaining complex concepts simply
- Use analogies when appropriate
- Be honest about limitations
- Focus on business value, not just technical details
