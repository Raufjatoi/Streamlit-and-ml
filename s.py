# Modeling

# Importing libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import streamlit as st

# Load data
data = pd.read_csv('goldstock.csv')

# Prepare data for modeling
# Assume 'X' is the independent variables and 'y' is the target variable (closing price)
X = data[['Open', 'High', 'Low', 'Volume']]  # Features: Open, High, Low prices, and Volume
y = data['Close']  # Target: Closing price

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Streamlit App
st.title("Gold Price Prediction Model")

st.write("## Model Evaluation Metrics")
st.write(f"Mean Squared Error: {mse}")
st.write(f"R^2 Score: {r2}")

# User input for prediction
st.write("## Predict Future Prices")
open_price = st.number_input('Open Price', value=0.0)
high_price = st.number_input('High Price', value=0.0)
low_price = st.number_input('Low Price', value=0.0)
volume = st.number_input('Volume', value=0)

# Predict button
if st.button('Predict'):
    input_data = np.array([[open_price, high_price, low_price, volume]])
    prediction = model.predict(input_data)
    st.write(f"Predicted Closing Price: ${prediction[0]:.2f}")
