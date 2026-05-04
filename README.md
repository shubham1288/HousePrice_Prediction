# 🏡 House Price Prediction with ML Pipeline

**Regression · EDA · Feature Engineering · Model Evaluation**

An end-to-end machine learning project to predict house prices using structured data. The focus is on building a clean pipeline, understanding the dataset, and clearly explaining decisions rather than chasing the highest accuracy.

---

## 📌 Project Overview

This project walks through a complete ML workflow:
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Model training and comparison
- Performance evaluation

The goal is to build a **readable, well-documented notebook** that demonstrates strong fundamentals.

---

## ⏱️ Project Details

- **Duration:** 5–6 Hours  
- **Date:** April 28, 2026  
- **Level:** Intermediate  
- **Total Points:** 60  

---

## 📂 Dataset Description

The dataset contains **1,500 rows** with property and location details.

### 🏠 Property Features
- `house_id` — Unique ID  
- `area_sqft` — Area in square feet  
- `bedrooms` — Number of bedrooms  
- `bathrooms` — Number of bathrooms  
- `floors` — Number of floors  
- `year_built` — Construction year  
- `garage` — 0 or 1  

### 📍 Location & Target
- `locality` — Urban / Suburban / Rural  
- `distance_to_city_km` — Distance from city center  
- `school_rating` — Score (1–10)  
- `renovated` — 0 or 1  
- `condition` — Poor / Average / Good / Excellent  
- `price_inr` — **Target variable (₹ in lakhs)**  

---

## ⚠️ Data Challenges

- ~3% missing values in:
  - `school_rating`
  - `distance_to_city_km`
- 8 duplicate rows  
- Mild outliers in `price_inr`  

---

## 🛠️ Tech Stack

- **Python**
- **Libraries:**
  - Pandas, NumPy
  - Matplotlib, Seaborn
  - Scikit-learn

---

## 🔄 Project Pipeline

### 1. Data Cleaning
- Handle missing values (imputation or removal)
- Remove duplicates
- Fix data types

### 2. Exploratory Data Analysis (EDA)
- Distribution of house prices
- Feature vs price relationships
- Correlation heatmap
- Identify key patterns

### 3. Feature Engineering & Preprocessing
- Create new features (e.g., house age)
- Encode categorical variables
- Scale numerical features
- Train-test split

### 4. Model Training
- Linear Regression
- Random Forest Regressor

### 5. Model Evaluation
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score
- Predicted vs Actual comparison plots

---

## 📈 Results & Insights

- Compared linear vs non-linear models
- Identified key price drivers such as:
  - Area (sqft)
  - Location (locality, distance)
  - Condition of property
- Final model selected based on performance and interpretability

---

## 📦 Live Project Link
https://shubham128-housepriceprediction.hf.space/
