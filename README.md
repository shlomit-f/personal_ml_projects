# MS Final Project - Sentiment Analysis
My final project for MS: Deep Learning vs. Traditional Machine Learning for Sentiment Analysis: A Performance Comparison 

Including:
1. EDA for text data and text pre-processing
2. ML predictions (NB, SVM, XGBoost)
3. DistilBERT pre-trained predictions (from Hugging Face)
4. DistilBERT pre-trained + finetune training predictions (including training process using pytorch)
   
# classic_ml
## Unsupervised Clustering

Implementation of kmeans and DB-scan:
1. Data scaling + feature selection
2. Parameters selection (k using silhouette and epsilon, min_points using KNN graph and silhouette)
3. Models' training
4. Results comparison including PCA visualisation, DB-scan score and simple counters.

## CTR prediction
Prediction of CTR:
1. EDA
2. Handeling missing values
3. Feature engineering
4. Scaling & Standartization
5. Handeling unbalanced dataset (ADASYN)
6. prediction models: KNN, XGBoost, Random Forest
7. Explainable AI - SHAP

## ARIMA Time Series Analysis
Manual fit of ARIMA models (including stationary validation)

# Statistics_projects
## Monte Carlo
This project is using monte carlo method for prediction of covid 19 trends.

The training data : Israel's new daily cases on 20.1.2022-20.3.2022

Target: Predicting the following 21 days (new daily cases) to the training data

Data origin: https://datadashboard.health.gov.il/COVID-19/general

# kaggle_competitions

## disaster_tweets - WIP
https://www.kaggle.com/competitions/nlp-getting-started/overview
Predict which Tweets are about real disasters and which ones are not

## contradictory-my-dear-watson - WIP
https://www.kaggle.com/competitions/contradictory-my-dear-watson/overview
Label each two sentences - are they entailment/neutral/contradiction.

# Visualisation
## Complex visualisation
1. Sankey diagram for multi-levels feature (sport.football,sport.basketball.nba)
2. Smoothing function
3. Confidence interval bar chart function - for AB testing

## Cheat sheet for plotly graphs
