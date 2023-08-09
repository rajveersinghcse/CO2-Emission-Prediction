[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://rajveersinghcse-co2emissionsprediction.streamlit.app/)
[![MIT LICENSE](https://badgen.net//badge/license/MIT/green)](https://github.com/rajveersinghcse/Reliance_Stock_Market_Prediction/blob/main/LICENSE)   ![MAINTAINED BADGE](https://img.shields.io/badge/Maintained%3F-yes-green.svg) 

# CO2 Emissions by Cars Predictions 

![Banner](https://github.com/rajveersinghcse/rajveersinghcse/blob/master/img/CO2_emissions.jpg)

<h3>Hey Folks,👨🏻‍💻</h3>
<p>I have created a project for <b>CO2 Emission Prediction</b>b> by vehicles that can predict the CO2 emission of a vehicle. Here I used the CO2 emission data of the car. I used that data to estimate how much CO2 a car is going to emit. I did this project during my internship.</p>

# Description of The Project:
<h3><b>Business Objective of the project</b></h3>

- The fundamental goal here is to model the CO2 emissions as a function of several car engine features.
- We have to use the data to estimate how much CO2 a car will emit.

# Description of The Data?
- I collected this data from the Canadian Government's Official [website](https://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64#wb-auto-6).

# About Data 📈 

- Make, car brand under study.
- Model, the specific model of the car.
- Vehicle_class, the car body type of the car.
- Engine_size, size of the car engine, in Litres.
- Cylinders, number of cylinders.
- Transmission, "A" for 'Automatic', "AM" for 'Automated manual', "AS" for 'Automatic with select shift', "AV" for 'Continuously variable', "M" for 'Manual'.
- Fuel_type, "X" for 'Regular gasoline', "Z" for 'Premium gasoline', "D" for 'Diesel', "E" for 'Ethanol (E85)', "N" for 'Natural gas'.
- Fuel_consumption_city, City fuel consumption ratings, in liters per 100 kilometers.
- Fuel_consumption_hwy, Highway fuel consumption ratings, in liters per 100 kilometers.
- Fuel_consumption_comb(l/100km), the combined fuel consumption rating (55% city, 45% highway), in L/100 km.
- Fuel_consumption_comb(mpg), the combined fuel consumption rating (55% city, 45% highway), in miles per gallon (mpg).
- Co2_emissions, the tailpipe emissions of carbon dioxide for combined city and highway driving, in grams per kilometer.


# Libraries and Language that I used in the project. 
<img height="25" width="80" src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54"> <img height="25" width="70" src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white"> <img height="25" width="80" src="https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black"> <img height="25" width="70" src="https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white"> <img height="25" width="110" src="https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white"> <img height="25" width="90" src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white"> 


## How to install these libraries?

### You can install these libraries by using the command.

- It can install all the libraries in your system which I have used in my project. 

- You will need Python in your system to use this command. You can use this given link to install Python in your system : [Python](https://www.python.org/downloads/)

- After installation of Python, you need to run this command in your command prompt.

```bash
pip install -r requirements.txt 
```
# Model Building.
- For the model building part, we used SVR, Random Forest, KNN, LSTM, and GRU models.

- I was getting more accuracy in LSTM than in other models. So I decided to use the LSTM model in my deployment program or main project.
<img height="170" width="350" src="https://github.com/rajveersinghcse/rajveersinghcse/blob/master/img/ModelBuilding.png" alt="ModelBuilding">

# Cloud version of this project.
- I deploy this project on the cloud you can check it out at this link: [Project](https://rajveersinghcse-reliance-stock-market-prediction-app-0xijl8.streamlit.app/)


# How to deploy the project?
- We used the Streamlit library for the deployment part of this project. To deploy or run this project in your local system, you must run this command in your command prompt.
```bash
streamlit run app.py 
```
---
<p align="center">
<b>Enjoy Coding</b>❤
</p>
