# 🚀 Fraud Detection Using PySpark  

## 📌 **Project Overview**  
This project aims to **detect fraudulent transactions** using machine learning models on a large dataset. The dataset consists of various transaction details such as amount, sender, receiver, and balance information. We preprocess the data, perform feature engineering, and build a predictive model to classify transactions as **fraudulent (1) or non-fraudulent (0)**.  

## 📂 **Dataset Description**  
The dataset (`fraud.csv`) contains the following columns:  

| Feature          | Description |
|-----------------|-------------|
| **step**        | Time step (1 step = 1 hour) |
| **type**        | Transaction type (CASH-IN, CASH-OUT, DEBIT, PAYMENT, TRANSFER) |
| **amount**      | Transaction amount |
| **nameOrig**    | Sender’s account ID |
| **oldbalanceOrg** | Sender’s balance before the transaction |
| **newbalanceOrig** | Sender’s balance after the transaction |
| **nameDest**    | Receiver’s account ID |
| **oldbalanceDest** | Receiver’s balance before the transaction (unknown for merchants) |
| **newbalanceDest** | Receiver’s balance after the transaction (unknown for merchants) |
| **isFraud**     | 1 = Fraudulent transaction, 0 = Legitimate transaction |
| **isFlaggedFraud** | 1 = Transaction flagged as fraud, 0 = Not flagged |

## 📊 **Data Preprocessing & Cleaning**  
- **Handle missing values** in recipient balances.  
- **Remove merchants** (since they are never fraudulent).  
- **Feature encoding** for categorical variables (`type`).  
- **Feature scaling** for numerical columns.  
- **Check for class imbalance** in fraud cases.  

## 🔍 **Exploratory Data Analysis (EDA)**  
- **Visualize transaction types vs. fraud occurrence.**  
- **Check distribution of transaction amounts.**  
- **Analyze sender & receiver balance patterns.**  
- **Identify anomalies in flagged fraudulent transactions.**  

## 🛠️ **Modeling Approach**  
- **Feature Selection:**  
  - Transaction amount, balance changes, transaction type, etc.  
  - Remove non-relevant columns (`nameOrig`, `nameDest`).  

- **Algorithms Used:**  
  ✅ Logistic Regression  
  ✅ Random Forest  
  ✅ Gradient Boosting
  ✅ Decision Trees
  
    

- **Performance Metrics:**  
  - **Precision** (False Positives matter more in fraud detection!)  
  - **Recall** (We don’t want to miss fraudulent cases!)  
  - **F1 Score** (Balances Precision & Recall)  
  - **ROC-AUC** (Overall model performance)  

## 📈 **Model Evaluation & Results**  
- **Compare models based on Precision, Recall, and F1-score.**  
- **Feature importance analysis to understand key fraud indicators.**  
- **Hyperparameter tuning to optimize performance.**  

## 🔐 **Fraud Prevention Strategies**  
- Implement **real-time monitoring systems** for high-value transactions.  
- Flag transactions with **suspicious balance behaviors**.  
- Use **anomaly detection** to detect unseen fraud patterns.  
- Enhance security measures like **two-factor authentication** for high-risk accounts.  

## 🚀 **How to Run the Project**  
### 1️⃣ **Set up the environment**  
```bash
pip install pyspark pandas matplotlib seaborn scikit-learn
```
### 2️⃣ **Run the script in Google Colab or Jupyter Notebook**  
```python
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("FraudDetection").getOrCreate()
```
### 3️⃣ **Load the dataset & process it**  
```python
df = spark.read.csv("fraud.csv", header=True, inferSchema=True)
```
### 4️⃣ **Train & evaluate models**  
```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

## 🏆 **Conclusion**  
- **The best-performing model was Random Forest ** with an **F1 score of 0.99**.  
- **Key fraud indicators:** Low transaction amount, transaction type.  
- **Future improvements:** Use **deep learning** & **real-time detection** with streaming data.  

