# 💼 AI-Powered Salary Management System

> Predict whether an employee earns more or less than ₹50,000/month using Machine Learning.

---

## 📌 Project Overview

This is a Streamlit-based web application that takes various demographic and employment-related inputs and uses a trained machine learning model to predict the **salary class** of an individual. The model is trained to classify whether the salary is **greater than ₹50K or less than ₹50K**.

It also supports **batch predictions** via CSV upload and showcases model performance metrics interactively.

---

## 🚀 Features

- 🎯 Predict individual salary class based on multiple inputs  
- 🗃️ Batch prediction support using CSV upload  
- 📊 Model performance metrics (Accuracy, Precision, Recall, F1 Score)  
- 💡 Clean and interactive UI built with **Streamlit**  
- 🔐 Encodes and maps inputs to match model training preprocessing  
- 📈 Visual bar chart of prediction probabilities

---

## 🧠 Technologies Used

- **Python**
- **Streamlit** – for building the web interface  
- **Pandas** – for data manipulation  
- **NumPy** – numerical operations  
- **Joblib** – model loading  
- **Scikit-learn** – for model training and prediction (assumed for saved model)

---

## 🏗️ Project Structure

salary-management-system/
│
├── best_model.pkl # Pre-trained classification model
├── app.py # Main Streamlit app
├── requirements.txt # Required dependencies
├── sample_input.csv # Sample input file for batch prediction
└── README.md # Project documentation



---

## ⚙️ How to Run the Project

### 🔧 Prerequisites

- Python 3.7+
- pip

### 🛠️ Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/itgithubplatform/salary_management_prediction_system.git
   cd salary-management-system

2.Install required dependencies:
pip install -r requirements.txt

3.Run the Streamlit app:
streamlit run app.py

