# Wellco Churn Prediction - Autoencoder Method

This notebook details the process of using an autoencoder-based anomaly detection method to identify users at high risk of churn. The process involves data loading, extensive preprocessing and feature engineering, model building and training, validation, and identification of potential churners among non-churned users.

## Process Overview

The following steps were performed:

1.  **Data Loading and Initial Exploration**:
    *   Loaded the `app_usage.csv`, `claims.csv`, `web_visits.csv`, and `churn_labels.csv` datasets.
    *   Performed initial checks on data shape, null values, and duplicates.
    *   Converted timestamp columns to datetime objects and extracted date components.
    *   Examined date ranges for each dataset.

2.  **Exploratory Data Analysis (EDA) and Preprocessing**:
    *   Analyzed app usage by visualizing the distribution of sessions per user.
    *   Explored the distribution of diagnosis codes in the claims data.
    *   Examined the distribution of web visit titles and descriptions.
    *   Engineered features in the `web_visits` data to identify content related to specific ICD-10 codes (E11.9, I10, Z71.3), general health-related content, and visits to "wellco" URLs.
    *   Calculated and visualized the churn rate and outreach rate from the `churn_labels`.
    *   Calculated the date difference between signup and the last app session for each user and visualized its distribution by churn status.

3.  **Feature Engineering - Creating the State Table**:
    *   Aggregated `claims`, `app_usage`, and `web_visits` data to create daily summaries per member (`agg_claims`, `agg_app_usage`, `agg_web_searches`).
    *   Created a comprehensive "state" table (`state_table`) with a row for each member for each day in a specified date range (2025-07-01 to 2025-07-14).
    *   Calculated days since signup for each member on each date in the state table.
    *   Left joined the aggregated daily data (`agg_claims`, `agg_app_usage`, `agg_web_searches`) to the `state_table`, filling missing values with 0.
    *   Engineered interaction features in the `state_table` to capture relationships between sessions, health-related searches, wellco searches, and diagnoses within a specific time window.
    *   Aggregated the `state_table` by `member_id` to create `agg_state_table`, with columns representing the average of daily features and interactions over the period.
    *   Merged churn status, outreach status, and days since signup into the `agg_state_table` to create the final `member_df` for modeling.

4.  **Autoencoder Model Validation**:
    *   Split the `member_df` into training, validation, and test sets based on percentiles of the 'days\_since\_signup' feature.
    *   Separated churned and non-churned users within the training data.
    *   Built and compiled an autoencoder model.
    *   Trained the autoencoder on the churned users from the training set.
    *   Calculated reconstruction errors for the churned training data.
    *   Identified and removed outlier churned users based on high reconstruction errors from the churned training set.
    *   Retrained the autoencoder on the filtered churned training data.
    *   Calculated reconstruction errors for the validation set (both churned and non-churned users).
    *   Visualized the distribution of reconstruction errors for churned and non-churned users in the validation set using stacked histograms.
    *   Used the KS test to find an optimal threshold for the reconstruction error based on the validation set to best separate churned and non-churned users.

5.  **Model Application and Identifying High-Risk Users**:
    *   Calculated reconstruction errors for the test set.
    *   Applied the determined threshold to the test set's reconstruction errors to make churn predictions.
    *   Evaluated the model's performance on the test set using a classification report and ROC AUC score.
    *   Calculated reconstruction errors for the non-churned users in the complete dataset.
    *   Identified non-churned users with no outreach who have reconstruction errors below the threshold (indicating churn-like behavior).
    *   Ranked these high-risk non-churned users by their reconstruction error.
    *   Saved the list of high-risk churn users.

This notebook demonstrates a method for identifying potential churners by training an autoencoder on known churned users and using the reconstruction error as an anomaly score to flag non-churned users who exhibit similar patterns.
