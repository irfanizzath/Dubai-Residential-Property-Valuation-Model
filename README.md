# Dubai Residential Property Valuation Model

An end-to-end machine learning project for estimating residential property prices in Dubai using historical transaction data from the Dubai Land Department.

This project builds and evaluates multiple regression models on more than 400,000 residential sales transactions between 2021–2025, with a final tuned LightGBM model achieving strong predictive performance.

---

## Overview

Dubai’s real estate market is highly dynamic, with property prices influenced by:

* Property size
* Location
* Number of rooms
* Market timing
* Proximity to key infrastructure
* Building characteristics

The goal of this project is to build a robust valuation model capable of predicting property transaction values from structured real estate transaction data.

The notebook covers the complete machine learning workflow:

* Data cleaning
* Exploratory data analysis
* Feature engineering
* Outlier treatment
* Encoding categorical variables
* Model training
* Hyperparameter tuning
* Diagnostic analysis
* Model serialization

---

## Dataset

**Source:** Dubai Land Department / Dubai Pulse Open Data Portal

Dataset contains historical Dubai property transaction records with:

* Transaction details
* Property characteristics
* Area/location information
* Property usage and transaction type
* Nearby landmarks
* Sales values

### Initial Dataset Size

* **Rows:** 1,661,875+
* **Columns:** 46

### Final Modelling Dataset

After preprocessing and filtering:

* Residential properties only
* Sales transactions only
* Years 2021–2025
* Outliers removed

Final training dataset:

* **~415,000 transactions**

---

## Problem Statement

Traditional property valuation is often:

* Time consuming
* Expert dependent
* Inconsistent across locations
* Difficult to scale

This project aims to automate residential property valuation using machine learning models trained on historical transaction data.

The model predicts:

* **Estimated transaction value (`actual_worth`)**

based on:

* Property size
* Area
* Rooms
* Transaction timing
* Nearby infrastructure
* Property type

---

## Tech Stack

### Languages & Libraries

* Python
* Pandas
* NumPy
* Scikit-learn
* LightGBM
* Optuna
* Matplotlib
* Seaborn
* Joblib

### Machine Learning Techniques

* Linear Regression baseline
* Gradient Boosting (LightGBM)
* Bayesian hyperparameter optimisation (Optuna)
* Target Encoding
* Log transformation
* Outlier clipping
* Cross-validation

---

## Data Preprocessing

### Key preprocessing steps include:

* Removing duplicate Arabic columns
* Standardising column names
* Missing value handling
* Date parsing and feature extraction
* Filtering sales transactions only
* Filtering residential properties only
* Room category consolidation
* Proximity feature engineering
* Log transforming skewed numerical variables
* Outlier treatment using percentile clipping

---

## Models

### 1. Linear Regression (Baseline)

### 2. LightGBM Regressor

### 3. Tuned LightGBM

Hyperparameters optimised using Optuna Bayesian optimisation.

---

## Model Performance

### Tuned LightGBM Performance

| Metric                | Score   |
| --------------------- | ------- |
| Validation R²         | ~0.883  |
| Strong Generalisation | Yes     |
| Overfitting           | Minimal |

The tuned LightGBM significantly outperformed the linear baseline and demonstrated stable validation performance.

---

## Key Findings

* Property size is the strongest valuation signal
* Location heavily influences pricing
* Transaction timing captures market cycles
* Log transformation greatly stabilises price variance
* LightGBM models nonlinear market behaviour effectively
* Target encoding improves high-cardinality location handling

---

## Repository Structure

```bash
.
├── Dubai Residential Property Valuation Model.ipynb        
├── model.joblib            
└── README.md
```

---

## Example Use Cases

This model can be extended for:

* Real estate valuation platforms
* Property recommendation systems
* Investment analysis
* Rental yield estimation
* Real estate analytics dashboards
* Automated valuation models (AVMs)

---

## Future Improvements

Potential future enhancements:

* Geospatial coordinate modelling
* Deep learning approaches
* Real-time API deployment
* Economic indicator integration
* Rental price prediction

---

## Results

The project demonstrates that modern gradient boosting methods can accurately model Dubai residential property prices using structured transaction data.

The final tuned LightGBM model achieved strong predictive performance while maintaining good generalisation across validation datasets.

---

## References

### Dataset

Dubai Land Department — Dubai Pulse Open Data Portal

[https://data.dubai.gov.ae/en/l/470061](https://data.dubai.gov.ae/en/l/470061)

---

## Author

**Mohamed Irfan**

M.Sc. AI & Computer Science Student

Data Analytics & Machine Learning Enthusiast
