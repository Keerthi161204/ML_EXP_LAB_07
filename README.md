# ML_EXP_LAB_07

# 📘 Ensemble Learning: Bagging, Boosting & Stacking

## 🔍 Overview

This project implements and compares three ensemble learning techniques:

* **Bagging (Bootstrap Aggregating)**
* **Boosting (AdaBoost)**
* **Stacking (Meta-Learning)**

The objective is to evaluate how different ensemble strategies improve model performance on a medical classification task.

---

## 🎯 Objectives

* Implement Bagging, Boosting, and Stacking models
* Apply **cross-validation** for reliable evaluation
* Compare ensemble methods using accuracy and classification metrics
* Understand bias-variance tradeoff in ensemble learning

---

## 📂 Dataset

* Dataset: **Breast Cancer Wisconsin Dataset** (Scikit-learn)
* Total samples: 569
* Features: 30 numerical features
* Classes:

  * `0` → Malignant
  * `1` → Benign

---

## ⚙️ Implementation Details

### 🔹 Data Handling

* Train-test split: 80% training, 20% testing
* Evaluation:

  * 5-fold Cross Validation
  * Test Accuracy
  * Classification Report

---

## 🧠 Models Used

### 🔸 Bagging

* Base estimator: Decision Tree
* Number of estimators: 50
* Max samples: 80%
* Key idea: Reduce **variance** by training multiple models on different subsets

---

### 🔸 Boosting (AdaBoost)

* Number of estimators: 100
* Learning rate: 0.5
* Key idea: Focus on **misclassified samples** iteratively

---

### 🔸 Stacking

* Base models:

  * Decision Tree
  * Gaussian Naive Bayes
  * Support Vector Machine
* Final model: Logistic Regression
* Key idea: Combine predictions using a **meta-learner**

---

## 📊 Results (Typical Performance)

| Model    | CV Accuracy | Test Accuracy |
| -------- | ----------- | ------------- |
| Bagging  | ~0.95–0.97  | ~0.95–0.97    |
| Boosting | ~0.95–0.98  | ~0.95–0.97    |
| Stacking | ~0.96–0.98  | ~0.96–0.98    |

---

## 📈 Key Observations

* Bagging improves stability by reducing variance
* Boosting improves performance by reducing bias
* Stacking combines strengths of multiple models → often best performance
* Cross-validation ensures results are not due to random splits

---

## 🧠 Conclusion

Ensemble methods significantly improve classification performance compared to single models. Among them, **stacking often performs best** because it leverages multiple learning algorithms and combines them intelligently.

---

## 🚀 How to Run

```bash id="l2m9k1"
pip install numpy scikit-learn
python your_script_name.py
```


