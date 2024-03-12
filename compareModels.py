# app.py

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv("OLDdata.csv")

# Remove the "Test Case" column
data.drop(columns=["Test Case"], inplace=True)

# Split data into features (X) and target (y)
X = data.drop(columns=["Final Premium"])
y = data["Final Premium"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
linear_model = LinearRegression()
gb_model = GradientBoostingRegressor()
rf_model = RandomForestRegressor()

# Train models
linear_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred_linear = linear_model.predict(X_test)
y_pred_gb = gb_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

# Evaluate models
mse_linear = mean_squared_error(y_test, y_pred_linear)
mse_gb = mean_squared_error(y_test, y_pred_gb)
mse_rf = mean_squared_error(y_test, y_pred_rf)

r2_linear = r2_score(y_test, y_pred_linear)
r2_gb = r2_score(y_test, y_pred_gb)
r2_rf = r2_score(y_test, y_pred_rf)

# Streamlit app
st.title("Insurance Premium Prediction App")
st.write("Enter feature values to predict the final premium:")

# Input fields for features (same as before)
base_rate = st.number_input("Base Rate", min_value=0)
employees = st.number_input("Employees", min_value=0)
state_mod_factor = st.number_input("State Mod Factor", min_value=0.0, max_value=2.0, step=0.01)
hazard_factor = st.number_input("Hazard Factor", min_value=0.0, max_value=2.0, step=0.01)
business_factor = st.number_input("Business Factor", min_value=0.0, max_value=2.0, step=0.01)
exp_mod_factor = st.number_input("Exp Mod Factor", min_value=0.0, max_value=2.0, step=0.01)
ilf_factor = st.number_input("ILF Factor", min_value=0.0, max_value=2.0, step=0.01)
deductible_factor = st.number_input("Deductible Factor", min_value=0.0, max_value=2.0, step=0.01)
discount_factor = st.number_input("Discount Factor", min_value=0.0, max_value=2.0, step=0.01)
minimum_premium = st.number_input("Minimum Premium", min_value=0)


# Predict button
if st.button("Predict"):
    # Create a DataFrame with user input (same as before)
    user_input = pd.DataFrame({
        "Base Rate": [base_rate],
        "Employees": [employees],
        "State Mod Factor": [state_mod_factor],
        "Hazard Factor": [hazard_factor],
        "Business Factor": [business_factor],
        "Exp Mod Factor": [exp_mod_factor],
        "ILF Factor": [ilf_factor],
        "Deductible Factor": [deductible_factor],
        "Discount Factor": [discount_factor],
        "Minimum Premium": [minimum_premium]
    })
    # Make predictions
    prediction_linear = linear_model.predict(user_input)
    prediction_gb = gb_model.predict(user_input)
    prediction_rf = rf_model.predict(user_input)

    st.success(f"Linear Regression Prediction: ${prediction_linear[0]:.2f}")
    st.success(f"Gradient Boosting Prediction: ${prediction_gb[0]:.2f}")
    st.success(f"Random Forest Prediction: ${prediction_rf[0]:.2f}")

# Display model evaluation metrics (same as before)
st.write(f"Linear Regression MSE: {mse_linear:.2f}, R-squared: {r2_linear:.2f}")
st.write(f"Gradient Boosting MSE: {mse_gb:.2f}, R-squared: {r2_gb:.2f}")
st.write(f"Random Forest MSE: {mse_rf:.2f}, R-squared: {r2_rf:.2f}")
