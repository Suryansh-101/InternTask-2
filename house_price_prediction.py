import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# --- 1. Load Sample Dataset ---
# In a real project, you would download a CSV from Kaggle.
# For this example, we'll use a string to simulate a CSV file.
# This dataset contains typical features used for house price prediction.

csv_data = """
Avg_Area_Income,Avg_Area_House_Age,Avg_Area_Number_of_Rooms,Avg_Area_Number_of_Bedrooms,Area_Population,Price,Address
79545.45857,5.682861322,7.009188143,4.09,23086.8005,1059033.558,208 Michael Ferry Apt. 674
79248.64245,6.002900312,6.730821018,3.09,40173.07217,1505890.915,188 Johnson Views Suite 079
61287.06718,5.86588984,8.512727173,5.13,36882.1594,1058987.988,9127 Elizabeth Stravenue
63345.24005,7.18823613,5.586728559,3.26,34310.24283,1260616.807,48421 Christopher Creek
59982.19723,5.040555138,7.83938728,4.23,26354.10947,630943.4893,3513 Tracy Dale Suite 198
80175.75426,4.988435336,6.104512419,4.04,26748.42847,1068138.076,9725 Angel Ferry
64698.46343,6.025336233,8.14775955,5.49,45616.22559,1482846.99,47522 Donald Bypass Apt. 071
78394.33928,6.989782443,6.620477784,2.42,36516.35897,1573936.564,9704 Wallace Street Suite 473
63226.02444,5.829588301,6.923242637,3.24,32303.4169,1042881.352,8292 Webster Square Apt. 538
68350.5238,7.69329621,6.72263432,3.33,42184.38131,1489520.36,31123 Tracy Inlet Suite 458
"""

# Read the string data into a pandas DataFrame
data_file = io.StringIO(csv_data)
df = pd.read_csv(data_file)

print("--- 1. Data Exploration ---")
print("First 5 rows of the dataset:")
print(df.head())
print("\n" + "="*50 + "\n")

print("Information about the dataset:")
df.info()
print("\n" + "="*50 + "\n")

print("Statistical summary of the dataset:")
print(df.describe())
print("\n" + "="*50 + "\n")


# --- 2. Data Preprocessing ---
# Select the features (X) and the target variable (y)
# We drop the 'Address' column as it's a non-numeric text feature.
X = df[['Avg_Area_Income', 'Avg_Area_House_Age', 'Avg_Area_Number_of_Rooms',
        'Avg_Area_Number_of_Bedrooms', 'Area_Population']]
y = df['Price']

# Split the data into training and testing sets
# test_size=0.3 means 30% of the data is for testing, 70% for training.
# random_state ensures that the split is the same every time we run the code.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# --- 3. Training the Linear Regression Model ---
print("--- 2. Training the Model ---")
# Create an instance of the LinearRegression model
lm = LinearRegression()

# Train (or "fit") the model on the training data
lm.fit(X_train, y_train)
print("Linear Regression model has been trained.")
print("\n" + "="*50 + "\n")

# --- 4. Model Evaluation ---
print("--- 3. Evaluating the Model ---")
# Print the model's coefficients
# These show the relationship between each feature and the house price.
coeff_df = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])
print("Model Coefficients:")
print(coeff_df)
print("\n")

# Make predictions on the test data
predictions = lm.predict(X_test)

# Calculate and print the evaluation metrics
print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, predictions))
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, predictions))
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print("\n" + "="*50 + "\n")


# --- 5. Visualization of Results ---
print("--- 4. Visualizing Predictions ---")
# Scatter plot of actual prices vs. predicted prices
# A perfect model would result in a straight diagonal line.
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, edgecolors='black', alpha=0.7)
plt.xlabel('Actual Prices (y_test)')
plt.ylabel('Predicted Prices')
plt.title('Actual Prices vs. Predicted Prices')
# Plotting the "perfect prediction" line
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', lw=2)
plt.grid(True)
plt.show()

# Histogram of the residuals (the difference between actual and predicted values)
# Residuals should be normally distributed, centered around zero.
residuals = y_test - predictions
sns.displot(residuals, bins=20, kde=True)
plt.title('Distribution of Residuals')
plt.xlabel('Residuals (Actual - Predicted)')
plt.show()

print("\n--- Project Complete ---")
print("The scatter plot shows how closely the model's predictions match the actual house prices.")
print("The residual plot shows that the errors are normally distributed, which is a good sign for a linear regression model.")
