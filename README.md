# 💳 Credit Card Fraud Detection System (Machine Learning)

## 📌 Project Overview
This project implements a Credit Card Fraud Detection System using Machine Learning in Python.
The model analyzes transaction data and predicts whether a transaction is Legitimate or Fraudulent
based on learned patterns from historical data.

The project is implemented fully in Jupyter Notebook and demonstrates a complete ML workflow
from data preprocessing to real-time prediction on unseen data.

---

## 🚀 Features
- Data preprocessing and cleaning
- Feature scaling using StandardScaler
- Handling highly imbalanced datasets
- Machine Learning model training and testing
- Real-time prediction for new transaction inputs
- Fraud / Legit transaction classification

---

## 🛠️ Technologies Used
- Python
- NumPy
- Pandas
- Scikit-learn
- Jupyter Notebook

---

## 📂 Project Structure
├── Credit Card Fraud Detection using Machine Learning in Python.ipynb
├── README.md

---

## 📊 Dataset Information
- Dataset consists of anonymized credit card transactions
- Features are PCA-transformed (V1–V28)
- Amount column represents transaction value
- Class column:
  - 0 → Legitimate Transaction
  - 1 → Fraudulent Transaction

---

## 🧠 Machine Learning Workflow
1. Data Loading
2. Data Exploration
3. Feature Scaling
4. Train-Test Split
5. Model Training
6. Model Evaluation
7. Prediction on New Transaction Data

---

## 🔮 How to Test a New Transaction

Provide input data in the same order as training features:

input_data = (
12, -2.79, -0.32, 1.64, 1.76, -0.13, 0.80, -0.42,
-1.90, 0.75, 1.15, 0.84, 0.79, 0.37, -0.73,
0.40, -0.30, -0.15, 0.77, 2.22, -1.58,
1.15, 0.22, 1.02, 0.02, -0.23, -0.23,
-0.16, -0.03, 58.8
)

Prediction Output:
- ✅ Legitimate Transaction
- 🚨 Fraudulent Transaction

---

## 📈 Model Evaluation
- Accuracy Score used for evaluation
- Model tested on unseen data
- Demonstrates real-world fraud detection use case

Note:
Due to data imbalance, accuracy alone is not sufficient.
Precision, Recall, and F1-score are important in real-world systems.

---

## ⚠️ Important Notes
- StandardScaler is fitted only on training data
- The same scaler is reused during prediction
- Model consistency is maintained during inference

---

## 🎯 Use Cases
- Credit card fraud detection
- Banking and financial systems
- Machine learning academic projects
- End-to-end ML workflow demonstration

---

## 📌 Future Improvements
- Batch prediction using CSV files
- Model deployment using Flask
- Advanced models (Random Forest, XGBoost)
- Performance optimization using ROC-AUC
- Dashboard visualization

---

## 👤 Author
Abhijit Mondal  
Aspiring Data Analyst / Machine Learning Enthusiast

---

## ⭐ Acknowledgement
This project is based on a publicly available anonymized credit card transaction dataset
used for academic and research purposes.
