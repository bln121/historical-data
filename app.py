import streamlit as st
from datetime import date

import yfinance as yf


import time
import datetime
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from datetime import date, timedelta



from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go


from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM



import base64
from IPython.display import HTML


#code 1


START="2010-01-01"
TODAY=date.today().strftime("%Y-%m-%d")
st.title("Stock Prediction App")

#Drop down to select symbol
#stocks=("SPY","GOOG","MSFT","GME")
#selected_stock=st.selectbox("Select dataset for prediction", stocks)

#symbol is given as user input
selected_stock=st.text_input('Enter symbol')

#button
result=st.button("Click Here")
#st.write(result)



n_years=st.slider("Years of prediction:", 1,4)
period=n_years*365

@st.cache_data
def load_data(ticker):
    data=yf.download(ticker,START, TODAY)
    data.reset_index(inplace=True)
    return data

time.sleep(30)

data_load_state=st.text("Load data...")
data=load_data(selected_stock)
data_load_state.text("Loading data...done!")

st.subheader('Raw data')
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

#Forecasting
df_train =data[['Date','Close']]
df_train= df_train.rename(columns={"Date":"ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader('Forecast data')
#for i in range(0,len(forecast)):
#fds=str(forecast.iloc[0][0]).split(" ")

#To print required 5 rows of forecast data

df_forecast=forecast.tail(370)
st.write(df_forecast.head())

st.write('forcast data')
fig1=plot_plotly(m,forecast)
st.plotly_chart(fig1)

st.write('forecast components')
fig2 = m.plot_components(forecast)
st.write(fig2)


#code 2 to print next day's close value



ticker = selected_stock
period1 = int(time.mktime(datetime.datetime(2021, 4, 1, 23, 59).timetuple()))#year,month,day
period2= int(time.mktime(datetime.datetime.now().timetuple()))
interval='1d' #1d, 1mo, 1wk

query_string =f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'
df=pd.read_csv(query_string)

data = pd.DataFrame(df)



for k in range(0,7):
  for i in range(0,2):
    if(i==0):
      open_close='Open'
    else:
      open_close='Close'
    #Prepare Data
    scaler=MinMaxScaler(feature_range=(0,1))
    scaled_data=scaler.fit_transform(data[open_close].values.reshape(-1,1))

    prediction_days = 30

    x_train=[]
    y_train=[]

    for x in range(prediction_days, len(scaled_data)-k):
        x_train.append(scaled_data[x-prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    #Build the Model
    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1)) #prediction of the next closing value

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=50, batch_size=32)

    #Test the model accuracy on existing data

    interval='1d' #1d, 1mo, 1wk

    test_data = pd.DataFrame(df)
    actual_prices=test_data[open_close].values

    total_dataset=pd.concat((data[open_close], test_data[open_close]), axis=0)

    model_inputs=total_dataset[len(total_dataset)-len(test_data)-prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)

    # Make Predictions on Test Data

    x_test=[]

    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x-prediction_days:x, 0])

    x_test=np.array(x_test)
    x_test=np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predicted_prices=model.predict(x_test)
    predicted_prices=scaler.inverse_transform(predicted_prices)

    # Plot the test predictions
    plt.plot(actual_prices, color = "black", label=f"Actual {ticker} Price")
    plt.plot(predicted_prices, color="green", label=f"Predicted {ticker} Price")
    plt.title(f"{ticker} Share Price")
    plt.xlabel("Time")
    plt.ylabel(f"{ticker} Share Price")
    plt.legend()
    #plt.show()

    #Predict Next Day

    real_data = [model_inputs[len(model_inputs)-prediction_days:len(model_inputs+1), 0]]
    real_data = np.array(real_data)
    real_data=np.reshape(real_data, (real_data.shape[0], real_data.shape[1],1))

    prediction=model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)

    if(k==0):
      if(i==0):
        prediction_open0=float(prediction)
        continue
      else:
        prediction_close0=float(prediction)
        continue

    if(k==1):
      if(i==0):
        prediction_open1=float(prediction)
        continue
      else:
        prediction_close1=float(prediction)
        continue
    
    if(k==2):
      if(i==0):
        prediction_open2=float(prediction)
        continue
      else:
        prediction_close2=float(prediction)
        continue

    if(k==3):
      if(i==0):
        prediction_open3=float(prediction)
        continue
      else:
        prediction_close3=float(prediction)
        continue

    if(k==4):
      if(i==0):
        prediction_open4=float(prediction)
        continue
      else:
        prediction_close4=float(prediction)
        continue

    if(k==5):
      if(i==0):
        prediction_open5=float(prediction)
        continue
      else:
        prediction_close5=float(prediction)
        continue

    if(k==6):
      if(i==0):
        prediction_open6=float(prediction)
        continue
      else:
        prediction_close6=float(prediction)
        

#NextDay_Date =str(datetime.datetime.today() + datetime.timedelta(days=1)).split()
#print(f"NextDay_Date: {NextDay_Date[0]}")

#Creating empty dataframe- actual_data

actual_data = pd.DataFrame(columns = ["Date","Open","prediction_open","accuracy_open","High","Low","Close","prediction_close","accuracy_close","actual_direction","prediction_direction","overall_direction","Adj Close","Volume"])

data1=data.tail(6)
actual_data=pd.merge(data1,actual_data,how='outer')
#actual_data=actual_data.append(data.tail(6),ignore_index=True)  #inserting last five rows from data1 into actual_data

actual_data.at[0,"prediction_open"]=round(prediction_open6,2)
actual_data.at[0,"prediction_close"]=round(prediction_close6,2)

actual_data.at[1,"prediction_open"]=round(prediction_open5,2)
actual_data.at[1,"prediction_close"]=round(prediction_close5,2)

actual_data.at[2,"prediction_open"]=round(prediction_open4,2)
actual_data.at[2,"prediction_close"]=round(prediction_close4,2)

actual_data.at[3,"prediction_open"]=round(prediction_open3,2)
actual_data.at[3,"prediction_close"]=round(prediction_close3,2)

actual_data.at[4,"prediction_open"]=round(prediction_open2,2)
actual_data.at[4,"prediction_close"]=round(prediction_close2,2)

actual_data.at[5,"prediction_open"]=round(prediction_open1,2)
actual_data.at[5,"prediction_close"]=round(prediction_close1,2)

#to calculate accuracy for historical data
for i in range(0,5):
  actual_data.at[i,"accuracy_open"]=100-abs((actual_data.at[i,"prediction_open"]-actual_data.at[i,"Open"])/actual_data.at[i,"Open"]*100)

for i in range(0,5):
  actual_data.at[i,"accuracy_close"]=100-abs((actual_data.at[i,"prediction_close"]-actual_data.at[i,"Close"])/actual_data.at[i,"Close"]*100)

#to insert actual_direction,prediction_direction,overall_direction

for i in range(0,6):
  if(abs(actual_data.at[i,'Close']-actual_data.at[i,'Open'])<=3 ):
    img_path='<img src="https://tse2.mm.bing.net/th?id=OIP.ddhO9ual65nyztsl1oxyVAFRC5&pid=Api&P=0&h=180" alt="Flat" width="50" height="40">'
    flag1=0
  elif(actual_data.at[i,'Close']-actual_data.at[i,'Open']>=3):
    img_path = '<img src="https://cdn1.iconfinder.com/data/icons/basic-ui-elements-coloricon/21/11-512.png" alt="Up" width="50" height="40">'
    flag1=1
  elif(actual_data.at[i,'Open']-actual_data.at[i,'Close']>=3):
    img_path='<img src="https://www.freeiconspng.com/uploads/red-arrow-png-26.png" alt="Down" width="50" height="40">'
    flag1=-1
  #with open(img_path, 'rb') as f:
   # img_bytes = f.read()
  #img_b64 = base64.b64encode(img_bytes).decode('utf-8')
  img_b64=img_path
# Add the image data to the DataFrame
  #actual_data.at[i,'actual_direction'] = '<img src="data:image/jpeg;base64,' + img_b64 + '" style="width:50%;height:40%; ">'
  actual_data.at[i,'actual_direction'] = img_b64
  if(abs(actual_data.at[i,'prediction_close']-actual_data.at[i,'prediction_open'])<=3):
    img_path='<img src="https://tse2.mm.bing.net/th?id=OIP.ddhO9ual65nyztsl1oxyVAFRC5&pid=Api&P=0&h=180" alt="Flat" width="50" height="40">'
    flag2=0
  elif(actual_data.at[i,'prediction_close']-actual_data.at[i,'prediction_open']>=3):
    img_path = '<img src="https://cdn1.iconfinder.com/data/icons/basic-ui-elements-coloricon/21/11-512.png" alt="Up" width="50" height="40">'
    flag2=1
  elif(actual_data.at[i,'prediction_open']-actual_data.at[i,'prediction_close']>=3):
    img_path='<img src="https://www.freeiconspng.com/uploads/red-arrow-png-26.png" alt="Down" width="50" height="40">'
    flag2=-1
  
  #with open(img_path, 'rb') as f:
  #  img_bytes = f.read()
  #img_b64 = base64.b64encode(img_bytes).decode('utf-8')
  img_b64=img_path
# Add the image data to the DataFrame
  #actual_data.at[i,'prediction_direction'] = '<img src="data:image/jpeg;base64,' + img_b64 + '" style="width:50%;height:20%; ">'
  actual_data.at[i,'prediction_direction'] = img_b64
#code to insert correct or wrong symbol

  if(flag1==flag2):
    img_path='<img src="https://tse3.mm.bing.net/th?id=OIP.oHwE7W6T_2kEtiaccChqAQHaHa&pid=Api&P=0&h=180 alt="Correct" width="50" height="40">'
  else:
    img_path='<img src="https://tse1.mm.bing.net/th?id=OIP.-HP-9rZqrTaXhXP-QV-nTwHaGk&pid=Api&P=0&h=180 alt="Wrong" width="50" height="40">'

  #with open(img_path, 'rb') as f:
#   img_bytes = f.read()
# img_b64 = base64.b64encode(img_bytes).decode('utf-8')
  img_b64=img_path
# Add the image data to the DataFrame
  #actual_data.at[i,'overall_direction'] = '<img src="data:image/jpeg;base64,' + img_b64 + '" style="width:50%;height:20%; ">'
  actual_data.at[i,'overall_direction'] = img_b64
# add 1 to each index
actual_data.index = actual_data.index + 1


#st.write(actual_data.head(5))

 #to print historical data -- first 5 rows of actual_data assigned to five_rows
st.subheader('Prediction of historical data')

st.write(HTML(actual_data.head(5).to_html(escape=False)))
st.write(actual_data.head(5))


future_data = pd.DataFrame(columns = ["Date","Open","prediction_open","accuracy_open","High","Low","Close","prediction_close","accuracy_close","actual_direction","prediction_direction","overall_direction","Adj Close","Volume"])

data2=actual_data.tail(1)
future_data=pd.merge(data2,future_data,how='outer')

#future_data = future_data.append(actual_data.tail(1),ignore_index=True)

#Accuracy for future data
future_data.at[0,"accuracy_open"]=100-abs((actual_data.at[6,"prediction_open"]-actual_data.at[6,"Open"])/actual_data.at[6,"Open"]*100)
future_data.at[0,"accuracy_close"]=100-abs((actual_data.at[6,"prediction_close"]-actual_data.at[6,"Close"])/actual_data.at[6,"Close"]*100)


future_data.at[1,"prediction_open"]=round(prediction_open0,2)
future_data.at[1,"prediction_close"]=round(prediction_close0,2)#Prediction tomorrow's value

#Converting string format date into date  and below is the code to insert the date in future_data dataframe

from datetime import datetime


date_str=future_data['Date'].iloc[0]
tomorrow = datetime.strptime(date_str, '%Y-%m-%d').date() + timedelta(1)
future_data['Date'].iloc[1] = tomorrow
future_data.index = future_data.index + 1
#st.write(future_data)


st.subheader("Prediction of future data")

st.write(HTML(future_data.to_html(escape=False)))
