# **Credit Card Fraud Detection 🛡️💳**

## **Overview**
This repository contains the implementation of a **neural network-based model** to detect fraudulent transactions in credit card data. The dataset is sourced from **[Kaggle's Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)**, which contains 284,807 transactions, of which 492 are fraudulent (highly imbalanced).

### **Goals**
- Accurately identify fraudulent transactions using a **binary classification model**.
- Address the **class imbalance problem** using techniques like **SMOTE** and **Random Undersampling**.
- Evaluate the model performance using metric **Recall**.

---

## **Dataset Preview**
| Feature           | Description                                                               |
|--------------------|---------------------------------------------------------------------------|
| `Time`            | Seconds elapsed between the transaction and the first transaction in the dataset. |
| `V1` to `V28`     | Principal components obtained via PCA (features are anonymized).         |
| `Amount`          | Transaction amount.                                                      |
| `Class`           | Target variable (0 = Non-fraudulent, 1 = Fraudulent).                   |

### **Key Characteristics**
- **Highly imbalanced dataset**: Only ~0.17% of the transactions are fraudulent.
- **Preprocessed data**: PCA-transformed features to preserve anonymity.

---

## **Results**
- **Model Architecture**: A **fully connected neural network** with the following layers:
  - Input normalization layer.
  - Dense layers with **ReLU activations** and **Dropout regularization**.
  - Sigmoid activation for binary classification.

- **Metrics Evaluated**:
  - **Recall (Fraudulent Transactions)**: 70%


### **Confusion Matrix**
![image](https://github.com/user-attachments/assets/4d7af1cc-f1ba-443e-b98b-7b43b097a9ca)

---

## **How to Clone and Run the Project**
Follow these steps to set up the project on your local machine:

### **1. Clone the Repository**
```bash
git clone https://github.com/nicole-baltodano/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### **2. Set Up the Environment**
Install the required dependencies:
```bash
pip install -r requirements.txt
```


### **4. Run the Project**
Run the pipeline using:
```bash
python main.py
```
