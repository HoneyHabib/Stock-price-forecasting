#Import the Libraries
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential , load_model
from tensorflow.keras.layers import LSTM , Dense
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#Get the stock quote
df = web.DataReader('TATASTEEL.NS',data_source='yahoo' , start='2010-01-01' , end='2021-05-05')
#Show the Data
#print(df)

#Get the no of rows and coulmns
#print(df.shape)

#Visualize the closing price

plt.figure(figsize=(12,8))
plt.title('HIGH PRICE HISTORY')
plt.plot((df['High']))
plt.xlabel('date' , fontsize=18)
plt.ylabel('High price in Rs.')
plt.show()

#Create a new Data frame only with 'high column'
data = df.filter(['High'])
#convert the Data frame into numpy array
dataset = data.values
#Get the number of rows to train the model on
training_data_len = round(len(dataset)*.8)
print(training_data_len)

#Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
print(scaled_data)

#Create the training dataset
#Create the sclaed training dataset
train_data = scaled_data[0:training_data_len, :]
#split the data into x_train and y_train
x_train = []
y_train = []

for i in range (60, len(train_data)):
  x_train.append(train_data[i-60:i-0])
  y_train.append(train_data[i,0])
  if i<= 61:
    print(x_train)
    print(y_train)
    print()

# Convert x_train and y_train to numpy array
x_train , y_train = np.array(x_train) , np.array(y_train)


#Reshape the data (but its already in 3-d )
print(x_train.shape)

#Build the LSTM model
# Define the Model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(60,1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

#Compile the model
model.compile(optimizer='adam' , loss='mean_squared_error')

#Train the model
#fitting the model
model.fit(x_train, y_train, batch_size=1 , epochs=1)

#Create the Testing data set with remaining rows
#create a new array containing scaled values from index 1348 to end of dataset
test_data = scaled_data[training_data_len - 60: , :]
#Create the Data set x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :] 
for i in range(60,len(test_data)):
  x_test.append(test_data[i-60:i,0])

#Convert the data into numpy array
x_test = np.array(x_test)

#reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

# Get the models predicted price values
predictions = model.predict(x_test)
#inverse transform the data i.e. unscaling the values
predictions = scaler.inverse_transform(predictions)

#save the model
model.save('Lstm_model.h5')

#Evaluate the model
#Get the root mean square error (RMSE)
rmse = np.sqrt(np.mean(predictions - y_test)**2)
print(rmse)

#Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['predictions'] = predictions


#VIsualize the data

plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('date', fontsize=18)
plt.ylabel('High price in Rs.')
plt.plot(train['High'])
plt.plot(valid[['High','predictions']])
plt.legend(['train', 'val', 'predictions'],loc ='upper left')
plt.show()

#show the valid and predicted prices
print(valid)

#Get the quote
steel_quote = web.DataReader('TATASTEEL.NS' , data_source='yahoo' , start='2012-01-01' , end='2021-05-05')
# Create new data frame
new_df = steel_quote.filter(['High'])
#get the last 60 days closing price values and covert the dataframe to an array
last_60_days = new_df[-60:].values
#scaling the data in (0,1)
last_60_days_scaled = scaler.transform(last_60_days)
#create an empty List
X_test = []
#Append the past 60 days 
X_test.append(last_60_days_scaled)
#Convert the x_test dataset into numpy array
X_test = np.array(X_test)
#Reshape the data
#X_test = np.reshape(X_test, (x_test.shape[0], X_test.shape[1], 1))
#Get the predicted scaled price
predicted_price = model.predict(X_test)
#unscaling
predicted_price = scaler.inverse_transform(predicted_price)
print(predicted_price) 
