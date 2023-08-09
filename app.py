import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import datetime
import streamlit as st
import model_building as m


with st.sidebar:
    st.markdown("# Reliance Stock Market Prediction")
    user_input = st.multiselect('Please select the stock',['RELIANCE.NS'],['RELIANCE.NS'])

    # user_input = st.text_input('Enter Stock Name', "ADANIENT.NS")
    st.markdown("### Choose Date for your anaylsis")
    START = st.date_input("From",datetime.date(2015, 1, 1))
    END = st.date_input("To",datetime.date(2023, 2, 28))
    bt = st.button('Submit') 

#adding a button
if bt:

# Importing dataset------------------------------------------------------
    df = yf.download('RELIANCE.NS', start=START, end=END)
    plotdf, future_predicted_values =m.create_model(df)
    df.reset_index(inplace = True)
    st.title('Reliance Stock Market Prediction')
    st.header("Data We collected from the source")
    st.write(df)

    reliance_1=df.drop(["Adj Close"],axis=1).reset_index(drop=True)
    reliance_2=reliance_1.dropna().reset_index(drop=True)

    reliance=reliance_2.copy()
    reliance['Date']=pd.to_datetime(reliance['Date'],format='%Y-%m-%d')
    reliance=reliance.set_index('Date')
    st.title('EDA')
    st.write(reliance)


# ---------------------------Graphs--------------------------------------

    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title('Visualizations')

    st.header("Graphs")
    plt.figure(figsize=(20,10))
    #Plot 1
    plt.subplot(2,2,1)
    plt.plot(reliance['Open'],color='green')
    plt.xlabel('Date')
    plt.ylabel('Open Price')
    plt.title('Open')
    #Plot 2
    plt.subplot(2,2,2)
    plt.plot(reliance['Close'],color='red')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title('Close')
    #Plot 3
    plt.subplot(2,2,3)
    plt.plot(reliance['High'],color='green')
    plt.xlabel('Date')
    plt.ylabel('High Price')
    plt.title('High')
    #Plot 4
    plt.subplot(2,2,4)
    plt.plot(reliance['Low'],color='red')
    plt.xlabel('Date')
    plt.ylabel('Low Price')
    plt.title('Low')
    st.pyplot()

#------------------------box-plots---------------------------------

    # Creating box-plots
    st.header("Box Plots")

    plt.figure(figsize=(20,10))
    #Plot 1
    plt.subplot(2,2,1)
    plt.boxplot(reliance['Open'])
    plt.xlabel('Date')
    plt.ylabel('Open Price')
    plt.title('Open')
    #Plot 2
    plt.subplot(2,2,2)
    plt.boxplot(reliance['Close'])
    plt.xlabel('Date')
    plt.ylabel('Cloes Price')
    plt.title('Close')
    #Plot 3
    plt.subplot(2,2,3)
    plt.boxplot(reliance['High'])
    plt.xlabel('Date')
    plt.ylabel('High Price')
    plt.title('High')
    #Plot 4
    plt.subplot(2,2,4)
    plt.boxplot(reliance['Low'])
    plt.xlabel('Date')
    plt.ylabel('Low Price')
    plt.title('Low')
    st.pyplot()

#---------------------Histogram---------------------------------------

    st.header("Histogram")
    # Ploting Histogram
    plt.figure(figsize=(20,10))
    #Plot 1
    plt.subplot(2,2,1)
    plt.hist(reliance['Open'],bins=50, color='green')
    plt.xlabel("Open Price")
    plt.ylabel("Frequency")
    plt.title('Open')
    #Plot 2
    plt.subplot(2,2,2)
    plt.hist(reliance['Close'],bins=50, color='red')
    plt.xlabel("Close Price")
    plt.ylabel("Frequency")
    plt.title('Close')
    #Plot 3
    plt.subplot(2,2,3)
    plt.hist(reliance['High'],bins=50, color='green')
    plt.xlabel("High Price")
    plt.ylabel("Frequency")
    plt.title('High')
    #Plot 4
    plt.subplot(2,2,4)
    plt.hist(reliance['Low'],bins=50, color='red')
    plt.xlabel("Low Price")
    plt.ylabel("Frequency")
    plt.title('Low')
    st.pyplot()


#-------------------------KDE Plots-----------------------------------------

    st.header("KDE Plots")
    # KDE-Plots
    plt.figure(figsize=(20,10))
    #Plot 1
    plt.subplot(2,2,1)
    sns.kdeplot(reliance['Open'], color='green')
    plt.title('Open')
    #Plot 2
    plt.subplot(2,2,2)
    sns.kdeplot(reliance['Close'], color='red')
    plt.title('Close')
    #Plot 3
    plt.subplot(2,2,3)
    sns.kdeplot(reliance['High'], color='green')
    plt.title('High')
    #Plot 4
    plt.subplot(2,2,4)
    sns.kdeplot(reliance['Low'], color='red')
    plt.title('Low')
    st.pyplot()


    st.header('Years vs Volume')
    st.line_chart(reliance['Volume'])


#-------------------Finding long-term and short-term trends---------------------

    st.title('Finding long-term and short-term trends')
    reliance_ma=reliance.copy()
    reliance_ma['30-day MA']=reliance['Close'].rolling(window=30).mean()
    reliance_ma['200-day MA']=reliance['Close'].rolling(window=200).mean()

    st.write(reliance_ma)


    st.subheader('Stock Price vs 30-day Moving Average')
    plt.plot(reliance_ma['Close'],label='Original data')
    plt.plot(reliance_ma['30-day MA'],label='30-MA')
    plt.legend()
    plt.title('Stock Price vs 30-day Moving Average')
    plt.xlabel('Date')
    plt.ylabel('Price')
    st.pyplot()


    st.subheader('Stock Price vs 200-day Moving Average')
    plt.plot(reliance_ma['Close'],label='Original data')
    plt.plot(reliance_ma['200-day MA'],label='200-MA')
    plt.legend()
    plt.title('Stock Price vs 200-day Moving Average')
    plt.xlabel('Date')
    plt.ylabel('Price')
    st.pyplot()

    df1 = pd.DataFrame(future_predicted_values)
    st.markdown("### Next 30 days forecast")
    df1.rename(columns={0: "Predicted Prices"}, inplace=True)
    st.write(df1)

    st.markdown("### Original vs predicted close price")
    fig= plt.figure(figsize=(20,10))
    sns.lineplot(data=plotdf)
    st.pyplot(fig)
    
    
else:
    #displayed when the button is unclicked
    st.write('Please click on the submit button to get the EDA ans Prediction') 
