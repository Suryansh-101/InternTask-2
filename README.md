# 🏡 House Price Prediction using Linear Regression

## 📌 Overview
This project was developed during my internship at **CodexIntern**.  
It demonstrates how to build and evaluate a **Linear Regression model** to predict **house prices** based on features such as:  

- Average area income  
- Average house age  
- Average number of rooms  
- Average number of bedrooms  
- Area population  

By training the model on a dataset (in practice, collected from Kaggle), we can estimate house prices and analyze feature importance.  

---

## 🚀 Features
- Load and preprocess a **housing dataset**  
- Train a **Linear Regression** model using `scikit-learn`  
- Evaluate model performance using metrics:  
  - **Mean Absolute Error (MAE)**  
  - **Mean Squared Error (MSE)**  
  - **Root Mean Squared Error (RMSE)**  
- Visualize results with:  
  - 📉 **Scatter Plot** of Actual vs Predicted Prices  
  - 📊 **Residual Distribution Plot**  

---

## 🛠️ Technologies Used
- **Python 3**  
- **Pandas** (data manipulation)  
- **NumPy** (numerical operations)  
- **Matplotlib & Seaborn** (visualizations)  
- **Scikit-learn** (machine learning model & evaluation)  

---

## 📈 Workflow
1. **Data Loading & Exploration**  
   - Load CSV dataset (Kaggle or custom sample)  
   - Explore data structure, summary statistics, and distributions  

2. **Data Preprocessing**  
   - Select features (`X`) and target (`y`)  
   - Drop irrelevant features (e.g., address)  
   - Split dataset into training (70%) and testing (30%) sets  

3. **Model Training**  
   - Train a `LinearRegression` model on training data  

4. **Model Evaluation**  
   - Print coefficients to interpret feature impact  
   - Compute error metrics (**MAE, MSE, RMSE**)  

5. **Visualization**  
   - Scatter plot of **Actual vs Predicted Prices**  
   - Residuals distribution to validate assumptions  

---

## ▶️ How to Run
1. Clone this repository:
   ```bash
   https://github.com/Suryansh-101/InternTask-2/blob/main/house_price_prediction.py

2. Install required libraries:
   ```bash
   pip install -r requirements.txt

3. Run the script:
   ```bash
   python house_price_prediction.py

---

## 📊 Example Outputs

1)Actual vs Predicted Scatter Plot → Shows how close predictions are to real values.

2)Residual Distribution Plot → Residuals are normally distributed around 0, validating regression assumptions.

---

👨‍💻 Developed during my internship at CODEXINTERN
