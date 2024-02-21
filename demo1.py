import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Sample dataset
data = {
    'Age': [35, 45, 28, 50, 60],
    'Location': ['Urban', 'Suburban', 'Rural', 'Urban', 'Suburban'],
    'Property Value': [250000, 400000, 150000, 600000, 700000],
    'Claims History': [0, 1, 2, 0, 3],
    'Weather Condition': ['Sunny', 'Rainy', 'Snowy', 'Sunny', 'Cloudy'],
    'Premium': [1000, 1500, 800, 2000, 2500]
}

df = pd.DataFrame(data)

# Encoding categorical variables
df = pd.get_dummies(df, columns=['Location', 'Weather Condition'])

# Splitting features and target variable
X = df.drop(columns=['Premium'])
y = df['Premium']

# Train a simple Random Forest Regressor model
model = RandomForestRegressor()
model.fit(X, y)

# Streamlit app
st.title('Insurance Premium Prediction')

# Input form for user
age = st.number_input('Enter your age', min_value=1, max_value=100, value=30)
location = st.selectbox('Select your location', ['Urban', 'Suburban', 'Rural'])
property_value = st.number_input('Enter your property value', min_value=0, value=100000)
claims_history = st.number_input('Enter your claims history', min_value=0, value=0)
weather_condition = st.selectbox('Select weather condition', ['Sunny', 'Rainy', 'Snowy', 'Cloudy'])

# Convert location and weather condition to one-hot encoding
location_encoded = [0, 0, 0]
if location == 'Urban':
    location_encoded[0] = 1
elif location == 'Suburban':
    location_encoded[1] = 1
else:
    location_encoded[2] = 1

weather_encoded = [0, 0, 0, 0]
if weather_condition == 'Sunny':
    weather_encoded[0] = 1
elif weather_condition == 'Rainy':
    weather_encoded[1] = 1
elif weather_condition == 'Snowy':
    weather_encoded[2] = 1
else:
    weather_encoded[3] = 1


# Create feature vector
input_data = {
    'Age': [age],
    'Property Value': [property_value],
    'Claims History': [claims_history],
    'Location_Urban': [location_encoded[0]],
    'Location_Suburban': [location_encoded[1]],
    'Location_Rural': [location_encoded[2]],
    'Weather Condition_Sunny': [weather_encoded[0]],
    'Weather Condition_Rainy': [weather_encoded[1]],
    'Weather Condition_Snowy': [weather_encoded[2]],
    'Weather Condition_Cloudy': [weather_encoded[3]]
}

# Get feature names from the training data
feature_names = X.columns.tolist()

# Create DataFrame with consistent feature names
input_df = pd.DataFrame(input_data, columns=feature_names)

# Predict premium
if st.button('Predict Premium'):
    prediction = model.predict(input_df)
    st.write('Predicted Premium:', prediction[0])

