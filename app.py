import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor


with st.sidebar:
    st.markdown("# CO2 Emissions by Vehical")
    user_input = st.selectbox('Please select',('Visulization','Model'))
    



# Load the vehicle dataset
df = pd.read_csv('co2 Emissions.csv')
#droping natural gas
df_natural = df[df["Fuel Type"].str.contains("N") == False].reset_index()
del df_natural['index']

# # we have to remove outliers from our data
df_new = df_natural[['Engine Size(L)','Cylinders','Fuel Consumption Comb (L/100 km)','CO2 Emissions(g/km)']]
df_new_model = df_new[(np.abs(stats.zscore(df_new)) < 1.9).all(axis=1)]


#-----------------------------Visulization----------------------------------
if user_input == 'Visulization':
    
    # we have to remove unwanted warnings from our program
    st.set_option('deprecation.showPyplotGlobalUse', False)

    # Showing Dataset ------------------------------------------------------
    st.title('CO2 Emissions by Vehical')
    st.header("Data We collected from the source")
    st.write(df)

#------------------------------- Barplot --------------------------------------
    st.header("EDA (Exploratory Data Analysis)")

    # Brands of Cars ---------------------------------------------------------------------
    st.subheader('Brands of Cars')
    df_brand = df['Make'].value_counts().reset_index().rename(columns={'count':'Count'})
    fig1 = plt.figure(figsize=(15, 6))
    figure1 = sns.barplot(data = df_brand, x = "Make",  y= "Count")
    plt.xticks(rotation = 75)
    plt.bar_label(figure1.containers[0])
    plt.title("All Car Companies and their Cars")
    plt.xlabel("Companies")
    plt.ylabel("Cars")
    st.pyplot(fig1)

    # Models of cars -----------------------------------------------------------------------
    st.subheader('Top 25 Models of cars')
    df_model = df['Model'].value_counts().reset_index().rename(columns={'count':'Count'})[:25]
    fig2 = plt.figure(figsize=(20,6))
    figure2 = sns.barplot(data = df_model, x = "Model",  y= "Count")
    plt.xticks(rotation = 75)
    plt.title("Top 25 Car Models")
    plt.xlabel("Models")
    plt.ylabel("Cars")
    plt.bar_label(figure2.containers[0])
    st.pyplot(fig2)

    # Vehical Class----------------------------------------------------------------------------
    st.subheader('Vehical Class')
    df_vehicle_class = df['Vehicle Class'].value_counts().reset_index().rename(columns={'count':'Count'})
    fig3 = plt.figure(figsize=(20,5))
    figure3 = sns.barplot(data = df_vehicle_class, x = "Vehicle Class",  y= "Count")
    plt.xticks(rotation = 75)
    plt.title("All Vehicle Class")
    plt.xlabel("Vehicle Class")
    plt.ylabel("Cars")
    plt.bar_label(figure3.containers[0])
    st.pyplot(fig3)


    # Engine Sizes of cars---------------------------------------------------------------------
    st.subheader('Engine Sizes of cars')
    df_engine_size = df['Engine Size(L)'].value_counts().reset_index().rename(columns={'count':'Count'})
    fig4 = plt.figure(figsize=(20,6))
    figure4 = sns.barplot(data = df_engine_size, x = "Engine Size(L)",  y= "Count")
    plt.xticks(rotation = 90)
    plt.title("All Engine Sizes")
    plt.xlabel("Engine Size(L)")
    plt.ylabel("Cars")
    plt.bar_label(figure4.containers[0])
    st.pyplot(fig4)

    # Cylinders--------------------------------------------------------------------------------
    st.subheader('Cylinders')
    df_cylinders = df['Cylinders'].value_counts().reset_index().rename(columns={'count':'Count'})
    fig5 = plt.figure(figsize=(20,6))
    figure5 = sns.barplot(data = df_cylinders, x = "Cylinders",  y= "Count")
    plt.xticks(rotation = 90)
    plt.title("All Cylinders")
    plt.xlabel("Cylinders")
    plt.ylabel("Cars")
    plt.bar_label(figure5.containers[0])
    st.pyplot(fig5)

    # Transmission of Cars------------------------------------------------------------------------
    st.subheader('Transmission of Cars')
    # Here we have to map similar labels into a single label for our Transmission column
    df["Transmission"] = np.where(df["Transmission"].isin(["A4", "A5", "A6", "A7", "A8", "A9", "A10"]), "Automatic", df["Transmission"])
    df["Transmission"] = np.where(df["Transmission"].isin(["AM5", "AM6", "AM7", "AM8", "AM9"]), "Automated Manual", df["Transmission"])
    df["Transmission"] = np.where(df["Transmission"].isin(["AS4", "AS5", "AS6", "AS7", "AS8", "AS9", "AS10"]), "Automatic with Select Shift", df["Transmission"])
    df["Transmission"] = np.where(df["Transmission"].isin(["AV", "AV6", "AV7", "AV8", "AV10"]), "Continuously Variable", df["Transmission"])
    df["Transmission"] = np.where(df["Transmission"].isin(["M5", "M6", "M7"]), "Manual", df["Transmission"])


    df_transmission = df['Transmission'].value_counts().reset_index().rename(columns={'count':'Count'})
    fig6 = plt.figure(figsize=(20,5))
    figure6 = sns.barplot(data = df_transmission, x = "Transmission",  y= "Count")
    plt.title("All Transmissions")
    plt.xlabel("Transmissions")
    plt.ylabel("Cars")
    plt.bar_label(figure6.containers[0])
    st.pyplot(fig6)

    # Fuel Type of Cars------------------------------------------------------------------------
    st.subheader('Fuel Type of Cars')
    df["Fuel Type"] = np.where(df["Fuel Type"]=="Z", "Premium Gasoline", df["Fuel Type"])
    df["Fuel Type"] = np.where(df["Fuel Type"]=="X", "Regular Gasoline", df["Fuel Type"])
    df["Fuel Type"] = np.where(df["Fuel Type"]=="D", "Diesel", df["Fuel Type"])
    df["Fuel Type"] = np.where(df["Fuel Type"]=="E", "Ethanol(E85)", df["Fuel Type"])
    df["Fuel Type"] = np.where(df["Fuel Type"]=="N", "Natural Gas", df["Fuel Type"])

    df_fuel_type = df['Fuel Type'].value_counts().reset_index().rename(columns={'count':'Count'})
    fig7 = plt.figure(figsize=(20,5))
    figure7 = sns.barplot(data = df_fuel_type, x = "Fuel Type",  y= "Count")
    plt.title("All Fuel Types")
    plt.xlabel("Fuel Types")
    plt.ylabel("Cars")
    plt.bar_label(figure7.containers[0])
    st.pyplot(fig7)
    st.text("We have only one data of Natural Gas. So we can Predicate anything by using only one data. That's why we have to drop this row.")
    
    # removing natural Gas------------------
    st.subheader('After removing Natural Gas data')
    df_new_fuel_type = df_natural['Fuel Type'].value_counts().reset_index().rename(columns={'count':'Count'})
    
    fig8 = plt.figure(figsize=(20,5))
    figure8 = sns.barplot(data = df_new_fuel_type, x = "Fuel Type",  y= "Count")
    plt.title("All Fuel Types")
    plt.xlabel("Fuel Types")
    plt.ylabel("Cars")
    plt.bar_label(figure8.containers[0])
    st.pyplot(fig8)

    
    # CO2 Emission variation with Brand ------------------------------------------------------------------------
    st.header('Variation in CO2 emissions with different features')
    st.subheader('CO2 Emission with Brand ')
    df_co2_make = df.groupby(['Make'])['CO2 Emissions(g/km)'].mean().sort_values().reset_index()
    fig8 = plt.figure(figsize=(20,5))
    figure8 = sns.barplot(data = df_co2_make, x = "Make",  y= "CO2 Emissions(g/km)")
    plt.xticks(rotation = 90)
    plt.title("CO2 Emissions variation with Brand")
    plt.xlabel("Brands")
    plt.ylabel("CO2 Emissions(g/km)")
    plt.bar_label(figure8.containers[0], fontsize=8, fmt='%.1f')
    st.pyplot(fig8)

    #  CO2 Emissions variation with Vehicle Class  ------------------------------------------------------------------------
    st.subheader('CO2 Emissions variation with Vehicle Class ')
    df_co2_vehicle_class = df.groupby(['Vehicle Class'])['CO2 Emissions(g/km)'].mean().sort_values().reset_index()
    fig9 = plt.figure(figsize=(23,5))
    figure9 = sns.barplot(data = df_co2_vehicle_class, x = "Vehicle Class",  y= "CO2 Emissions(g/km)")
    plt.xticks(rotation = 90)
    plt.title("CO2 Emissions variation with Vehicle Class")
    plt.xlabel("Vehicle Class")
    plt.ylabel("CO2 Emissions(g/km)")
    plt.bar_label(figure9.containers[0], fontsize=9)
    st.pyplot(fig9)

    
    # CO2 Emission variation with Transmission ------------------------------------------------------------------------
    st.subheader('CO2 Emission variation with Transmission ')
    df_co2_transmission = df.groupby(['Transmission'])['CO2 Emissions(g/km)'].mean().sort_values().reset_index()
    fig10 = plt.figure(figsize=(23,5))
    figure10 = sns.barplot(data = df_co2_transmission, x = "Transmission",  y= "CO2 Emissions(g/km)")
    plt.xticks(rotation = 90)
    plt.title("CO2 Emissions variation with Transmission")
    plt.xlabel("Transmission")
    plt.ylabel("CO2 Emissions(g/km)")
    plt.bar_label(figure10.containers[0], fontsize=10)
    st.pyplot(fig10)

    
    # CO2 Emissions variation with Fuel Type ------------------------------------------------------------------------
    st.subheader('CO2 Emissions variation with Fuel Type')
    df_co2_fuel_type = df.groupby(['Fuel Type'])['CO2 Emissions(g/km)'].mean().sort_values().reset_index()
    fig11 = plt.figure(figsize=(23,5))
    figure11 = sns.barplot(data = df_co2_fuel_type, x = "Fuel Type",  y= "CO2 Emissions(g/km)")
    plt.xticks(rotation = 90)
    plt.title("CO2 Emissions variation with Fuel Type")
    plt.xlabel("Fuel Type")
    plt.ylabel("CO2 Emissions(g/km)")
    plt.bar_label(figure11.containers[0], fontsize=10)
    st.pyplot(fig11)

    
    #------------------------box-plots---------------------------------

    # Creating box-plots
    st.header("Box Plots")

    plt.figure(figsize=(20,10))
    #Plot 1
    plt.subplot(2,2,1)
    plt.boxplot(df_new['Engine Size(L)'])
    plt.title('Engine Size(L)')
    #Plot 2
    plt.subplot(2,2,2)
    plt.boxplot(df_new['Cylinders'])
    plt.title('Cylinders')
    #Plot 3
    plt.subplot(2,2,3)
    plt.boxplot(df_new['Fuel Consumption Comb (L/100 km)'])
    plt.title('Fuel Consumption Comb (L/100 km)')
    #Plot 4
    plt.subplot(2,2,4)
    plt.boxplot(df_new['CO2 Emissions(g/km)'])
    plt.title('CO2 Emissions(g/km)')
    st.pyplot()

    # Outliers ---------------------------------------------------------------------
    st.text("As we can see there are some outliers present in our Dataset")
    st.subheader("After removing outliers")
    st.write("Before removing ouliers we have",len(df),"data")
    st.write("After removing ouliers we have",len(df_new_model),"data")

    
    # Creating new box-plots-------------------------------------------------------------
    st.subheader("Boxplot after removing outliers")
    plt.figure(figsize=(20,10))
    #Plot 1
    plt.subplot(2,2,1)
    plt.boxplot(df_new_model['Engine Size(L)'])
    plt.title('Engine Size(L)')
    #Plot 2
    plt.subplot(2,2,2)
    plt.boxplot(df_new_model['Cylinders'])
    plt.title('Cylinders')
    #Plot 3
    plt.subplot(2,2,3)
    plt.boxplot(df_new_model['Fuel Consumption Comb (L/100 km)'])
    plt.title('Fuel Consumption Comb (L/100 km)')
    #Plot 4
    plt.subplot(2,2,4)
    plt.boxplot(df_new_model['CO2 Emissions(g/km)'])
    plt.title('CO2 Emissions(g/km)')
    st.pyplot()




else:
    # Prepare the data for modeling
    X = df_new_model[['Engine Size(L)','Cylinders','Fuel Consumption Comb (L/100 km)']]
    y = df_new_model['CO2 Emissions(g/km)']

    # Train the linear regression model
    model = RandomForestRegressor()
    model.fit(X, y)

    # Create the Streamlit web app
    st.title('CO2 Emission Prediction')
    st.write('Enter the vehicle specifications to predict CO2 emissions.')

    # Input fields for user
    engine_size = st.number_input('Engine Size(L)',step=0.1, format="%.1f")
    cylinders = st.number_input('Cylinders', min_value=2, max_value=16, step=1)
    fuel_consumption = st.number_input('Fuel Consumption Comb (L/100 km)',step=0.1, format="%.1f")

    # Predict CO2 emissions
    input_data = [[cylinders, engine_size, fuel_consumption]]
    predicted_co2 = model.predict(input_data)

    # Display the prediction
    st.write(f'Predicted CO2 Emissions: {predicted_co2[0]:.2f} g/km')