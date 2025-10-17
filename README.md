#### Name - Yashraj Ramdas Chavan  
#### Roll No. - DA25M031  
#### Assignment 6 - Data Analytics Lab (DA5401) JUL–NOV 2025  

# 🧾 DA5401 A6 — Imputation via Regression for Missing Data

---

### 📘 Overview
This assignment explores various **data imputation techniques** and evaluates how each impacts the performance of a **Logistic Regression classifier**.  
Using the **UCI Credit Card Default Dataset**, we simulate missing values under a *Missing At Random (MAR)* assumption and compare three imputation approaches — **Median**, **Linear Regression**, and **KNN Regression** — against **Listwise Deletion**.

---

### 🎯 Objective
To assess the effectiveness of different imputation strategies by:
- Introducing artificial missing values into key numeric columns.  
- Imputing them using multiple methods (Median, Linear, KNN).  
- Comparing classification performance across the cleaned datasets.

---

### 🧩 Workflow Summary

#### **Part A — Data Preprocessing & Imputation**
1. Loaded the UCI Credit Card dataset.  
2. Introduced **10% Missing At Random (MAR)** values in `AGE`, `BILL_AMT1`, and `BILL_AMT2`.  
3. Created four datasets:
   - **Dataset A:** Median Imputation (Baseline).  
   - **Dataset B:** Linear Regression Imputation for `AGE`.  
   - **Dataset C:** KNN Regression Imputation for `AGE`.  
   - **Dataset D:** Listwise Deletion (removed all rows with missing data).

#### **Part B — Model Training & Evaluation**
- Standardized features using `StandardScaler`.  
- Trained a **Logistic Regression** model on each dataset.  
- Evaluated models using **Accuracy**, **Precision**, **Recall**, and **F1-score** metrics.  

#### **Part C — Comparative Analysis**
- Compiled performance metrics for all models.  
- Created comparative bar plots (color-blind–friendly).  
- Discussed results and recommended the best imputation method.

---

### 📊 Key Results

| Dataset | Method | Accuracy | F1 (Class 1) | Macro Avg F1 | Weighted Avg F1 | Observation |
|:---------|:---------|:---------:|:-------------:|:--------------:|:----------------:|:-------------|
| **A** | Median Imputation | 0.8078 | 0.3526 | 0.6199 | 0.7689 | Simplest, ignores feature relationships. |
| **B** | Linear Regression | 0.8204 | 0.3891 | 0.6419 | 0.7830 | Better — captures linear dependencies. |
| **C** | KNN Regression | 0.8210 | 0.3925 | 0.6437 | 0.7840 | Best — handles non-linear relations. |
| **D** | Listwise Deletion | 0.8165 | 0.3780 | 0.6352 | 0.7786 | Acceptable but loses data (reduced generalization). |

---

### 🧠 Conclusion
- **Imputation > Deletion:** Keeping all data using imputation provides more stable and generalizable models.  
- **KNN Regression Imputation** slightly outperformed others, demonstrating that **non-linear imputations** are effective when feature relationships are complex.  
- **Linear Regression Imputation** also improved model performance but assumes linearity.  
- **Median Imputation** is a quick baseline but should not be used in serious predictive tasks.  
- **Listwise Deletion** should only be used when missingness is minimal.

**✅ Recommended Approach:**  
Use **KNN Regression Imputation** for this dataset and similar real-world scenarios — it balances interpretability, performance, and robustness.

---

### ⚙️ Requirements
**Python Version:** 3.8 or higher  
**Libraries Used:**
```bash
pandas
numpy
scikit-learn
matplotlib
