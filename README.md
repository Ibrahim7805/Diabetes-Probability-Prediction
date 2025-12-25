# 🩺 Diabetes Probability Prediction
**Kaggle Playground Series S5E12 | Dual-Pipeline Approach for Risk Assessment**

---

## 📌 Overview
This repository presents a comprehensive machine learning system designed to predict the probability of diabetes diagnosis. The project explores two distinct data environments:
1. **Original Data:** Real-world health indicators for robust clinical baseline modeling.
2. **Synthetic Data:** High-fidelity generated data for the Kaggle Playground competition.

The objective is to maximize the **ROC-AUC Score** while maintaining model interpretability across both datasets.

---

## 📂 Project Structure

### 🏢 [01. Original Data Model](./01-Original-Data-Model)
*Focused on real-world patterns and deployment.*
* **Data Source:** [Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/mohankrishnathalla/diabetes-health-indicators-dataset)
* **Components:** * `EDA`: In-depth analysis of 50+ health features.
    * `Preprocessing`: Pipeline for handling real-world data noise.
    * `Supervised ML`: Tuned XGBoost model for baseline prediction.
    * `Streamlit`: Interactive web app for real-time risk assessment.

### 🧪 [02. Synthetic Data Model](./02-Synthetic-Data-Model)
*Focused on competition performance (In Progress).*
* **Competition:** [Kaggle Playground Series S5E12](https://www.kaggle.com/competitions/playground-series-s5e12)
* **Strategy:** Handling synthetic artifacts and feature distribution shifts to optimize ROC-AUC for the public/private leaderboard.

---

## 🛠️ Technical Implementation
* **Language:** Python 🐍
* **Models:** XGBoost, LightGBM (Gradient Boosting).
* **Evaluation:** Area Under the ROC Curve (ROC-AUC).
* **Deployment:** Streamlit Dashboard.

---

## 📊 Key Insights (EDA Highlights)
* Analysis of feature distributions (Body Mass Index, HighBP, Smoker) between original and synthetic datasets.
* Correlation analysis to identify the top drivers of diabetes risk.

---

## 🚀 Future Roadmap
- [x] Complete Original Data Pipeline & Streamlit App.
- [ ] Finalize Synthetic Data Feature Engineering.
- [ ] Submit final predictions to Kaggle Leaderboard.
- [ ] Perform Model Stacking for higher accuracy.

---

## 👤 Contact Me
**Ibrahim Ashraf** [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ibrahim-ashraf-968a65338)  
[![Facebook](https://img.shields.io/badge/Facebook-1877F2?style=for-the-badge&logo=facebook&logoColor=white)](https://www.facebook.com/ebrahim.ashraf.7805)
