# Titanic - Machine Learning from Disaster 🚢

My first Kaggle machine learning project! This repository contains a complete, robust machine learning pipeline built locally in VS Code to predict survival on the Titanic.

## 🛠️ Tech Stack & Workflow
* **Language**: Python
* **Libraries**: Pandas, Scikit-Learn, XGBoost, NumPy
* **Environment**: Local VS Code development

## 🚀 Key Features & Approach
1. **Feature Engineering**: Handling missing values and data preprocessing.
2. **Model**: **XGBoost Classifier** (fine-tuned with regularization parameters like `max_depth=3` and `learning_rate=0.05` to prevent overfitting).
3. **Validation**: **5-Fold Cross-Validation (`StratifiedKFold` / `KFold`)** for robust and reliable local performance evaluation without data leakage.

## 📁 Files in this Repository
* `main.py`: The main script handling data preprocessing, cross-validation, and generating `submission.csv`.