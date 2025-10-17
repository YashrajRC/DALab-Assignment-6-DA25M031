#### Name - Yashraj Ramdas Chavan
#### Roll no. - DA25M031
#### Assignment 6 - Data Analytics Lab ( Da5401) JUL-NOV 2025 

# DA5401 A6 — Imputation via Regression for Missing Data

### 📘 Overview
This assignment explores different techniques for handling missing data and studies how each method impacts the performance of a classification model.  
Using the **UCI Credit Card Default Dataset**, we simulate missing values and compare multiple imputation strategies before training a **Logistic Regression** model.

---

### 🎯 Objective
To evaluate how various imputation methods — **Median**, **Linear Regression**, and **KNN Regression** — affect the quality of data and the resulting model performance, compared to **Listwise Deletion**.

---

### 🧩 Workflow Summary

#### **Part A — Data Preprocessing & Imputation**
1. Loaded the UCI Credit Card dataset.  
2. Introduced artificial **Missing At Random (MAR)** values (10%) in numerical columns `AGE`, `BILL_AMT1`, and `BILL_AMT2`.  
3. Created four datasets:
   - **Dataset A:** Median imputation (baseline).  
   - **Dataset B:** Linear Regression imputation for `AGE`.  
   - **Dataset C:** KNN Regression imputation for `AGE`.  
   - **Dataset D:** Listwise deletion (rows with NaN removed).

#### **Part B — Model Training & Evaluation**
- Standardized all features using `StandardScaler`.  
- Trained a **Logistic Regression** classifier on each dataset.  
- Evaluated models using **Accuracy, Precision, Recall, and F1-score**.  
- Compared results to see which imputation method led to the best classification performance.

#### **Part C — Comparative Analysis**
- Summarized results in a table and plotted F1-scores for visual comparison.  
- Discussed the trade-off between deletion and imputation, explained the difference between linear and non-linear regression imputations, and recommended the most effective method.

---

### 📊 Key Findings
| Model | Method | F1-score | Observation |
|--------|---------|-----------|--------------|
| A | Median Imputation | 0.35 | Simplest, but ignores relationships. |
| B | Linear Regression | 0.39 | Better, captures feature relationships. |
| C | KNN Regression | 0.39+ | Best overall, handles non-linear patterns. |
| D | Listwise Deletion | 0.37 | Performs okay but loses data and reduces generalization. |

---

### 🧠 Conclusion
- Imputation methods that use feature relationships (Linear or KNN) outperform simple median filling and listwise deletion.  
- **KNN Regression Imputation** slightly outperformed others, showing that **non-linear models handle complex relationships better**.  
- For real-world datasets, it’s best to use regression-based imputation instead of deleting rows.

---

### ⚙️ Requirements
Python 3.8+  
Libraries used:
```bash
pandas
numpy
scikit-learn
matplotlib

