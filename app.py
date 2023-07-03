import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Load the vehicle dataset
df = pd.read_csv('co2 Emissions.csv')

# Prepare the data for modeling
X = df[['Engine Size(L)','Cylinders','Fuel Consumption Comb (L/100 km)']]
y = df['CO2 Emissions(g/km)']

# Train the linear regression model
model = RandomForestRegressor()
model.fit(X, y)

# Create the Streamlit web app
st.title('CO2 Emission Prediction')
st.write('Enter the vehicle specifications to predict CO2 emissions.')

# Input fields for user
engine_size = st.number_input('Engine Size(L)',step=0.1,format="%.1f")
cylinders = st.number_input('Cylinders', min_value=2, max_value=16, step=1)
fuel_consumption = st.number_input('Fuel Consumption Comb (L/100 km)',step=0.1,format="%.1f")

# Predict CO2 emissions
input_data = [[cylinders, engine_size, fuel_consumption]]
predicted_co2 = model.predict(input_data)

# Display the prediction
st.write(f'Predicted CO2 Emissions: {predicted_co2[0]:.2f} g/km')