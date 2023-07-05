# Importing libraries-----------------------------------------------------------------------------------------
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor

# Creating Sidebar-------------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("# CO2 Emissions by Vehicle")
    user_input = st.selectbox('Please select',('Visulization','Model'))

# Load the vehicle dataset
df = pd.read_csv('co2 Emissions.csv')

# Drop rows with natural gas as fuel type
fuel_type_mapping = {"Z": "Premium Gasoline","X": "Regular Gasoline","D": "Diesel","E": "Ethanol(E85)","N": "Natural Gas"}
df["Fuel Type"] = df["Fuel Type"].map(fuel_type_mapping)
df_natural = df[~df["Fuel Type"].str.contains("Natural Gas")].reset_index(drop=True)

# Remove outliers from the data
df_new = df_natural[['Engine Size(L)', 'Cylinders', 'Fuel Consumption Comb (L/100 km)', 'CO2 Emissions(g/km)']]
df_new_model = df_new[(np.abs(stats.zscore(df_new)) < 1.9).all(axis=1)]

# Visulization-------------------------------------------------------------------------------------------------
if user_input == 'Visulization':

    # Remove unwanted warnings---------------------------------------------------------------------------------
    st.set_option('deprecation.showPyplotGlobalUse', False)

    # Showing Dataset------------------------------------------------------------------------------------------
    st.title('CO2 Emissions by Vehicle')
    st.header("Data We collected from the source")
    st.write(df)

    # Brands of Cars-------------------------------------------------------------------------------------------
    st.subheader('Brands of Cars')
    df_brand = df['Make'].value_counts().reset_index().rename(columns={'count':'Count'})
    plt.figure(figsize=(15, 6))
    fig1 = sns.barplot(data=df_brand, x="Make", y="Count")
    plt.xticks(rotation=75)
    plt.title("All Car Companies and their Cars")
    plt.xlabel("Companies")
    plt.ylabel("Cars")
    plt.bar_label(fig1.containers[0], fontsize=7)
    st.pyplot()
    st.write(df_brand)

    # Top 25 Models of Cars------------------------------------------------------------------------------------
    st.subheader('Top 25 Models of Cars')
    df_model = df['Model'].value_counts().reset_index().rename(columns={'count':'Count'})
    plt.figure(figsize=(20, 6))
    fig2 = sns.barplot(data=df_model[:25], x="Model", y="Count")
    plt.xticks(rotation=75)
    plt.title("Top 25 Car Models")
    plt.xlabel("Models")
    plt.ylabel("Cars")
    plt.bar_label(fig2.containers[0])
    st.pyplot()
    st.write(df_model)

    # Vehicle Class--------------------------------------------------------------------------------------------
    st.subheader('Vehicle Class')
    df_vehicle_class = df['Vehicle Class'].value_counts().reset_index().rename(columns={'count':'Count'})
    plt.figure(figsize=(20, 5))
    fig3 = sns.barplot(data=df_vehicle_class, x="Vehicle Class", y="Count")
    plt.xticks(rotation=75)
    plt.title("All Vehicle Class")
    plt.xlabel("Vehicle Class")
    plt.ylabel("Cars")
    plt.bar_label(fig3.containers[0])
    st.pyplot()
    st.write(df_vehicle_class)

    # Engine Sizes of Cars-------------------------------------------------------------------------------------
    st.subheader('Engine Sizes of Cars')
    df_engine_size = df['Engine Size(L)'].value_counts().reset_index().rename(columns={'count':'Count'})
    plt.figure(figsize=(20, 6))
    fig4 = sns.barplot(data=df_engine_size, x="Engine Size(L)", y="Count")
    plt.xticks(rotation=90)
    plt.title("All Engine Sizes")
    plt.xlabel("Engine Size(L)")
    plt.ylabel("Cars")
    plt.bar_label(fig4.containers[0])
    st.pyplot()
    st.write(df_engine_size)

    # Cylinders-----------------------------------------------------------------------------------------------
    st.subheader('Cylinders')
    df_cylinders = df['Cylinders'].value_counts().reset_index().rename(columns={'count':'Count'})
    plt.figure(figsize=(20, 6))
    fig5 = sns.barplot(data=df_cylinders, x="Cylinders", y="Count")
    plt.xticks(rotation=90)
    plt.title("All Cylinders")
    plt.xlabel("Cylinders")
    plt.ylabel("Cars")
    plt.bar_label(fig5.containers[0])
    st.pyplot()
    st.write(df_cylinders)

    # Transmission of Cars------------------------------------------------------------------------------------
    transmission_mapping = { "A4": "Automatic", "A5": "Automatic", "A6": "Automatic", "A7": "Automatic", "A8": "Automatic", "A9": "Automatic", "A10": "Automatic", "AM5": "Automated Manual", "AM6": "Automated Manual", "AM7": "Automated Manual", "AM8": "Automated Manual", "AM9": "Automated Manual", "AS4": "Automatic with Select Shift", "AS5": "Automatic with Select Shift", "AS6": "Automatic with Select Shift", "AS7": "Automatic with Select Shift", "AS8": "Automatic with Select Shift", "AS9": "Automatic with Select Shift", "AS10": "Automatic with Select Shift", "AV": "Continuously Variable", "AV6": "Continuously Variable", "AV7": "Continuously Variable", "AV8": "Continuously Variable", "AV10": "Continuously Variable", "M5": "Manual", "M6": "Manual", "M7": "Manual"}
    df["Transmission"] = df["Transmission"].map(transmission_mapping)
    st.subheader('Transmission')
    df_transmission = df['Transmission'].value_counts().reset_index().rename(columns={'count': 'Count'})
    fig6 = plt.figure(figsize=(20, 5))
    sns.barplot(data=df_transmission, x="Transmission", y="Count")
    plt.title("All Transmissions")
    plt.xlabel("Transmissions")
    plt.ylabel("Cars")
    plt.bar_label(plt.gca().containers[0])
    st.pyplot(fig6)
    st.write(df_transmission)

    # Fuel Type of Cars--------------------------------------------------------------------------------------
    st.subheader('Fuel Type')
    df_fuel_type = df['Fuel Type'].value_counts().reset_index().rename(columns={'count': 'Count'})
    fig7 = plt.figure(figsize=(20, 5))
    sns.barplot(data=df_fuel_type, x="Fuel Type", y="Count")
    plt.title("All Fuel Types")
    plt.xlabel("Fuel Types")
    plt.ylabel("Cars")
    plt.bar_label(plt.gca().containers[0])
    st.pyplot(fig7)
    st.text("We have only one data on natural gas. So we cannot predict anything using only one data. That's why we have to drop this row.")
    st.write(df_fuel_type)

    # Removing Natural Gas-----------------------------------------------------------------------------------
    st.subheader('After removing Natural Gas data')
    df_ftype = df_natural['Fuel Type'].value_counts().reset_index().rename(columns={'count': 'Count'})
    fig8 = plt.figure(figsize=(20, 5))
    sns.barplot(data=df_ftype, x="Fuel Type", y="Count")
    plt.title("All Fuel Types")
    plt.xlabel("Fuel Types")
    plt.ylabel("Cars")
    plt.bar_label(plt.gca().containers[0])
    st.pyplot(fig8)
    st.write(df_ftype)

    # CO2 Emission variation with Brand----------------------------------------------------------------------
    st.header('Variation in CO2 emissions with different features')
    st.subheader('CO2 Emission with Brand ')
    df_co2_make = df.groupby(['Make'])['CO2 Emissions(g/km)'].mean().sort_values().reset_index()
    fig8 = plt.figure(figsize=(20, 5))
    sns.barplot(data=df_co2_make, x="Make", y="CO2 Emissions(g/km)")
    plt.xticks(rotation=90)
    plt.title("CO2 Emissions variation with Brand")
    plt.xlabel("Brands")
    plt.ylabel("CO2 Emissions(g/km)")
    plt.bar_label(plt.gca().containers[0], fontsize=8, fmt='%.1f')
    st.pyplot(fig8)

    def plot_bar(data, x_label, y_label, title):
        plt.figure(figsize=(23, 5))
        sns.barplot(data=data, x=x_label, y=y_label)
        plt.xticks(rotation=90)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.bar_label(plt.gca().containers[0], fontsize=9)

    # CO2 Emissions variation with Vehicle Class-------------------------------------------------------------
    st.subheader('CO2 Emissions variation with Vehicle Class')
    df_co2_vehicle_class = df.groupby(['Vehicle Class'])['CO2 Emissions(g/km)'].mean().sort_values().reset_index()
    plot_bar(df_co2_vehicle_class, "Vehicle Class", "CO2 Emissions(g/km)", "CO2 Emissions variation with Vehicle Class")
    st.pyplot()

    # CO2 Emission variation with Transmission---------------------------------------------------------------
    st.subheader('CO2 Emission variation with Transmission')
    df_co2_transmission = df.groupby(['Transmission'])['CO2 Emissions(g/km)'].mean().sort_values().reset_index()
    plot_bar(df_co2_transmission, "Transmission", "CO2 Emissions(g/km)", "CO2 Emission variation with Transmission")
    st.pyplot()

    # CO2 Emissions variation with Fuel Type--------------------------------------------------------------
    st.subheader('CO2 Emissions variation with Fuel Type')
    df_co2_fuel_type = df.groupby(['Fuel Type'])['CO2 Emissions(g/km)'].mean().sort_values().reset_index()
    plot_bar(df_co2_fuel_type, "Fuel Type", "CO2 Emissions(g/km)", "CO2 Emissions variation with Fuel Type")
    st.pyplot()

    # Box Plots-------------------------------------------------------------------------------------------
    st.header("Box Plots")
    plt.figure(figsize=(20, 10))
    features = ['Engine Size(L)', 'Cylinders', 'Fuel Consumption Comb (L/100 km)', 'CO2 Emissions(g/km)']
    for i, feature in enumerate(features, start=1):
        plt.subplot(2, 2, i)
        plt.boxplot(df_new[feature])
        plt.title(feature)
    st.pyplot()

    # Outliers-------------------------------------------------------------------------------------------
    st.text("As we can see there are some outliers present in our Dataset")
    st.subheader("After removing outliers")
    st.write("Before removing outliers we have", len(df), "data")
    st.write("After removing outliers we have", len(df_new_model), "data")

    # Boxplot after removing outliers-------------------------------------------------------------------
    st.subheader("Boxplot after removing outliers")
    plt.figure(figsize=(20, 10))
    for i, feature in enumerate(features, start=1):
        plt.subplot(2, 2, i)
        plt.boxplot(df_new_model[feature])
        plt.title(feature)
    st.pyplot()




else:
    # Prepare the data for modeling--------------------------------------------------------------------
    X = df_new_model[['Engine Size(L)', 'Cylinders', 'Fuel Consumption Comb (L/100 km)']]
    y = df_new_model['CO2 Emissions(g/km)']

    # Train the random forest regression model---------------------------------------------------------
    model = RandomForestRegressor().fit(X, y)

    # Create the Streamlit web app---------------------------------------------------------------------
    st.title('CO2 Emission Prediction')
    st.write('Enter the vehicle specifications to predict CO2 emissions.')

    # Input fields for user----------------------------------------------------------------------------
    engine_size = st.number_input('Engine Size(L)', step=0.1, format="%.1f")
    cylinders = st.number_input('Cylinders', min_value=2, max_value=16, step=1)
    fuel_consumption = st.number_input('Fuel Consumption Comb (L/100 km)', step=0.1, format="%.1f")

    # Predict CO2 emissions----------------------------------------------------------------------------
    input_data = [[cylinders, engine_size, fuel_consumption]]
    predicted_co2 = model.predict(input_data)

    # Display the prediction---------------------------------------------------------------------------
    st.write(f'Predicted CO2 Emissions: {predicted_co2[0]:.2f} g/km')
