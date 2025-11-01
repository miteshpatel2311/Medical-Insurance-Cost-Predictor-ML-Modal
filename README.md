# 🏥 Medical Insurance Cost Prediction using Machine Learning

## 📝 Overview

The objective of this project is to develop and evaluate regression models to **predict individual medical insurance costs** (`charges`) based on key demographic and lifestyle factors. Accurate cost prediction is vital for insurance companies to conduct risk assessment and helps individuals understand factors influencing their healthcare expenses.

This repository contains the dataset, the full analysis notebook (`Medical Insurance Cost Prediction.ipynb`), and the final trained models.

---

## 🚀 Project Objectives

The core goal is to **build and evaluate regression models** using the Scikit-Learn library to predict the `charges` value.

The complete machine learning workflow followed includes:

* Data loading & exploration
* Preprocessing & encoding
* Feature scaling
* Model training & evaluation
* Model deployment (saving models)

---

## 📊 Dataset & Features

The dataset comprises demographic and lifestyle information of insured individuals alongside their annual medical charges.

| Feature | Type | Description |
|:---------|:------|:-------------|
| `age` | Numeric | Age of the individual |
| `sex` | Categorical | Gender (`male`, `female`) |
| `bmi` | Numeric | Body Mass Index (measure of body fat) |
| `children` | Numeric | Number of dependents covered by insurance |
| `smoker` | Categorical | Smoking status (`yes`, `no`) |
| `region` | Categorical | Residential region in the U.S. |
| `charges` | Numeric (Target) | Individual medical insurance cost (in USD) |

**Dataset Source:** [Kaggle](https://www.kaggle.com/datasets/rahulvyasm/medical-insurance-cost-prediction)

---

## 🛠️ Methodology and Preprocessing

### 1. Data Cleaning
The dataset was loaded and explored, confirming there were **2772 rows and 7 columns** with **no missing (null) values**.

### 2. Feature Encoding
Categorical features were converted to a numeric format since Scikit-Learn algorithms require numerical input.
* `sex` and `smoker` were converted to binary numeric variables (e.g., `smoker`: `yes` $\rightarrow$ `1`, `no` $\rightarrow$ `0`).
* `region` (four categories) was converted using an arbitrary mapping/Ordinal Encoding.

### 3. Feature Scaling
**`StandardScaler`** was applied to the numeric input features (`age`, `bmi`, `children`) to standardize them to a $\mu=0$ and $\sigma=1$ distribution.
* Scaling is crucial for distance-based and gradient-based models (**Linear Regression**, **SVR**, **KNN**) as it prevents features with large ranges (like `charges`, `age`, `bmi`) from dominating the learning process.
* The target variable (`charges`) was **not scaled**.

### 4. Train-Test Split
The data was split into training and testing sets with a **20% test size** (`test_size=0.2`).

---

## 📈 Model Performance

Two regression models were trained and evaluated using **R-squared ($R^2$) score** and **Mean Squared Error (MSE)**.

| Model | $R^2$ Score (Test) | MSE (Test) |
|:---|:---|:---|
| **Linear Regression** | 0.7395 | 39,975,040.36 |
| **Random Forest Regressor** | **0.9509** | **7,540,070.42** |

**Conclusion:** The **Random Forest Regressor** demonstrated significantly better performance, achieving an $R^2$ score of approximately **0.95**, indicating it is the superior model for predicting medical insurance costs in this project.

---

## 💾 Saved Artifacts

The best-performing model (`rf_model`) and the necessary preprocessing tool (`scaler`) were saved using Python's `pickle` library for deployment and future inference.

* `rf_model.pkl` (Trained Random Forest Regressor)
* `scaler.pkl` (Trained StandardScaler)

---

## ⚙️ How to Run the Project

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/YourUsername/YourRepoName.git](https://github.com/YourUsername/YourRepoName.git)
    cd YourRepoName
    ```
2.  **Install Dependencies:**
    (Assumes Python 3 is installed)
    ```bash
    pip install pandas numpy scikit-learn
    ```
3.  **Run the Notebook:**
    Open the `Medical Insurance Cost Prediction.ipynb` notebook in Jupyter or VS Code to view the full step-by-step analysis, preprocessing, and model training.
