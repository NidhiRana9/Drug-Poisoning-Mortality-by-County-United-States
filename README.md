💊 Drug-Poisoning-Mortality-by-County-United-States
📝 Overview

This project focuses on predicting drug poisoning mortality rates across U.S. counties using Machine Learning techniques. The analysis leverages demographic and regional factors to understand patterns and predict death rates.

📊 Dataset

Source: NCHS - Drug Poisoning Mortality by County, United States

Description:

 Age-adjusted death rates for drug poisoning per 100,000 population.

 Deaths classified using ICD-10 codes for unintentional, suicide, homicide, and undetermined poisoning.

 Demographic features include state, county, urban/rural category, and census division.

 Project Steps
1️⃣ Data Cleaning & Preprocessing

 Renamed and formatted columns for consistency.

 Encoded categorical features using LabelEncoder.

 Removed leakage columns (lower_confidence_limit, upper_confidence_limit, standard_deviation).

2️⃣ Exploratory Data Analysis (EDA)

 Distribution plots with custom colors.

 Trend analysis of death rates over the years.

 Correlation heatmap for feature-target relationships.

3️⃣ Modeling

🏋️ Split dataset into training and test sets (80/20).

🤖 Trained three models:

Linear Regression

Random Forest Regressor

Gradient Boosting Regressor

📊 Evaluated performance using R², MAE, RMSE.

4️⃣ Feature Importance & Error Analysis

 Visualized top 10 features influencing death rates.

 Analyzed prediction errors with distribution plots.

🏆 Key Results
Model	R² Score
Linear Regression	0.280
Random Forest Regressor	0.881 ✅
Gradient Boosting Regressor	0.613

✅ Random Forest gave the best predictive performance.


