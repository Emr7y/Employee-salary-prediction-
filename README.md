
# 🧠 Employee Salary Prediction (Kaggle – Thapar Summer School)

## 📌 Project Overview
This repository contains my solution for the **Thapar Summer School – Employee Salary Prediction** Kaggle competition.  
The task is to **predict the salary** of employees based on their demographic and company information.  

- **Type:** Regression  
- **Target:** `salary` (continuous variable)  
- **Metric:** Mean Absolute Error (MAE)  
- **Best Model:** LightGBM (MAE ~11.6k on validation)  

---

## 📂 Repository Structure
```
├── notebooks/           # Kaggle/Colab notebooks
├── app.py               # Streamlit demo app
├── submission.csv       # Sample Kaggle submission
├── requirements.txt     # Dependencies
└── README.md            # Project documentation
```

---

## 📊 Approach
1. **Exploratory Data Analysis (EDA)**
   - Checked missing values and distributions  
   - Heatmap of correlations  
   - Boxplots of salary by categorical features  

2. **Feature Engineering**
   - `age_diff = age - age_when_joined`  
   - `bonus_per_year = annual_bonus / (years_in_the_company+1)`  
   - `total_experience = prior_years_experience + years_in_the_company`  
   - `experience_ratio = prior_years_experience / (years_in_the_company+1)`  
   - `company_department = company + "_" + department`  
   - `employment_type` = argmax of (`full_time`, `part_time`, `contractor`)  

3. **Modeling**
   - Benchmarked: RandomForest, XGBoost, LightGBM, Ridge, ElasticNet  
   - Compared using **Validation MAE**  
   - **LightGBM** gave the best performance (~11.6k MAE)  

4. **Submission**
   - Predictions aligned with the provided `sample_submission.csv` format  

---

## 📈 Results
| Model       | Val RMSE   | Val MAE   |
|-------------|------------|-----------|
| LightGBM    | ~14,678    | **11,648** |
| XGBoost     | ~14,686    | 11,724    |
| RandomForest| ~15,227    | 12,159    |
| Ridge       | ~15,897    | 12,779    |
| ElasticNet  | ~15,994    | 12,888    |

✅ **LightGBM was selected for the final submission.**

---

## 🚀 Streamlit App
A simple Streamlit app (`app.py`) is included to demonstrate model predictions interactively.  
Run locally with:

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## ❤️ Credits
Made with ❤️ by Emr7y | Model: LightGBM | Data: [Kaggle Competition](https://www.kaggle.com/competitions/thapar-summer-school-employee-salary-prediction)
