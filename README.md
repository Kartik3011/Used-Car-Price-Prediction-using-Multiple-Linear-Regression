# 🚗 Used Car Price Prediction using Multiple Linear Regression

An end-to-end regression pipeline built in Python to predict vehicle resale prices (`selling_price`) on the CarDekho dataset (~15,400 listings) using vehicle specifications, usage metrics, and brand features.

---

## 📌 Project Overview

Predicting fair market value for pre-owned cars is essential for both buyers and automotive dealerships. This project explores the predictive power and interpretability of **Multiple Linear Regression (MLR)** alongside **L1-regularized Lasso Regression**, addressing challenges such as right-skewed price distributions, multi-collinearity, and high-cardinality categorical variables.

---

## 📊 Dataset Summary

- **Source:** [CarDekho Used Car Dataset](https://www.kaggle.com/datasets/manishkr1754/cardekho-used-car-data)
- **Observations:** 15,411 rows
- **Target Variable:** `selling_price`
- **Key Features:**
  - **Numerical:** `vehicle_age`, `km_driven`, `mileage`, `engine`, `max_power`, `seats`
  - **Categorical:** `brand`, `model`, `seller_type`, `fuel_type`, `transmission_type`

---

## 🔍 Key EDA & Domain Insights

1. **Target Distribution Skewness:**
   - Median selling price is ₹5.56 Lakh, while maximum listings reach up to ₹3.95 Crore (e.g., luxury listings such as Ferrari, Rolls-Royce).
   - This heavy right-tail creates challenges for ordinary least squares (OLS) regression, which can yield negative price predictions for low-value vehicles.
2. **Feature Collinearity:**
   - `engine` capacity and `max_power` display strong positive correlation with each other and are the strongest drivers of vehicle price.
3. **Redundancy Removal:**
   - Dropped `car_name` prior to model fitting because it simply concatenates `brand` and `model`, introducing redundant cardinality without extra signal.

---

## 🛠️ Data Preprocessing & Pipeline Architecture

To prevent data leakage, all transformations are learned strictly on the training set (`fit_transform`) and applied downstream to the test set (`transform`):

```
                   Raw Dataset (15,411 rows)
                               │
                80/20 Train-Test Split (random_state=42)
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
     Numerical Features                    Categorical Features
 (age, km, engine, power, etc.)        (brand, model, fuel, etc.)
            │                                     │
      StandardScaler()                     CountEncoder()
            │                                     │
            │                              StandardScaler()
            └──────────────────┬──────────────────┘
                               ▼
                   ColumnTransformer Pipeline
                               ▼
                  Model Fitting & Prediction
```

- **Numerical Features:** Scaled using `StandardScaler` to ensure standardized variance.
- **Categorical Features:** Handled via `CountEncoder` (from `category_encoders`) to cleanly compress high-cardinality columns (`brand`, `model`) without creating an excessively wide one-hot matrix, followed by `StandardScaler`.

---

## ⚙️ Models & Performance

Models were evaluated using an 80/20 train-test split:

| Model | Split | MAE (₹) | $R^2$ Score |
| :--- | :--- | :--- | :--- |
| **Multiple Linear Regression (MLR)** | Train | ~263,490 | 0.625 |
| | **Test** | **~274,064** | **0.670** |
| **Lasso Regression ($\alpha = 0.1$)** | Train | ~263,490 | 0.625 |
| | **Test** | **~274,064** | **0.670** |

### Observations:
- **Baseline Accuracy:** Both models capture roughly **67% of price variance** ($R^2 \approx 0.67$) on unseen data.
- **Regularization Effect:** At $\alpha = 0.1$, Lasso yields nearly identical results to standard OLS, indicating that the count-encoded features do not suffer from severe overfitting in this configuration.

---

## 📦 Tech Stack

- **Language:** Python 3.10+
- **Data Manipulation:** `pandas`, `numpy`
- **Machine Learning:** `scikit-learn`, `category_encoders`
- **Visualization:** `matplotlib`, `seaborn`

---

## 🔮 Future Improvements

- [ ] Will apply **log transformation** ($\log(1 + y)$) to `selling_price` to stabilize variance and eliminate negative predictions.
- [ ] Experiment with polynomial feature expansion to capture non-linear depreciation curves.
- [ ] Benchmark against tree-based ensembles (Random Forest, XGBoost) to model complex feature interactions.
