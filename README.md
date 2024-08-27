# Stock Price Prediction with LSTM

Welcome to the **Stock Price Prediction with LSTM** repository! This project demonstrates how to use Long Short-Term Memory (LSTM) networks to predict the highest price of stocks based on historical data.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Model Building](#model-building)
  - [Evaluation and Visualization](#evaluation-and-visualization)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Project Overview

This repository contains a Python-based implementation of an LSTM model designed to forecast the highest price of a given stock. By analyzing historical stock price data, the model predicts future peak values, which can be valuable for traders and analysts looking to make informed decisions.

## Features

- **LSTM Model**: Advanced neural network architecture tailored for time-series prediction.
- **Data Preprocessing**: Includes steps for cleaning and preparing stock price data.
- **Performance Metrics**: Evaluate the accuracy of predictions with relevant metrics.
- **Visualization Tools**: Compare predicted values with actual prices using charts and graphs.

## Technologies Used

- **Python**: Main programming language used for implementation.
- **TensorFlow/Keras**: Libraries for building and training the LSTM model.
- **Pandas**: For data manipulation and preprocessing.
- **Matplotlib**: For visualizing results and performance.

## Getting Started

To get started with this project, follow these steps:

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/stock-price-prediction-lstm.git
   ```

2. **Navigate to the Project Directory**

   ```bash
   cd stock-price-prediction-lstm
   ```

3. **Install Dependencies**

   You will need to install the following Python packages:

   - TensorFlow
   - Keras
   - Pandas
   - Matplotlib
   - Seaborn

   You can install these packages using `pip`. Run the following command:

   ```bash
   pip install tensorflow keras pandas matplotlib
   ```

4. **Prepare Your Data**

   Ensure you have historical stock price data in the required format. Refer to the `data` folder for sample data or format your data accordingly.

5. **Run the Notebook**

   Open `Stock_Price_Prediction_LSTM.ipynb` using Jupyter Notebook or JupyterLab and follow the instructions to train and test the model.

## Usage

### Data Preparation

The first step is to load and preprocess the stock price data. Here’s a snippet of how the data is prepared:

```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load data
df = web.DataReader('TATASTEEL.NS',data_source='yahoo' , start='2010-01-01' , end='2021-05-05')

# Convert to DataFrame
data = df.filter(['High'])
```
# Normalize features
```python
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
```

### Model Building

Next, we build and train the LSTM model:

```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Define model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(60,1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error')

# Prepare training data
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

# Train model
model.fit(X_train, y_train, epochs=1, batch_size=1)
```

### Evaluation and Visualization

After training, evaluate the model and visualize the predictions:

```python
import matplotlib.pyplot as plt

# Make predictions
predictions = model.predict(x_test)

# Plot results
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('date', fontsize=18)
plt.ylabel('High price in Rs.')
plt.plot(train['High'])
plt.plot(valid[['High','predictions']])
plt.legend(['train', 'val', 'predictions'],loc ='upper left')
plt.show()
```
![stock price prediction graph](https://github.com/user-attachments/assets/900f5203-af3e-4e51-aaa3-b985f2aad78a)

## Contributing

Contributions are welcome! If you have suggestions, improvements, or fixes, please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or feedback, please reach out to [your email address] or open an issue in the repository.

Happy coding! 🚀

---

Feel free to customize or add more detailed snippets based on the specific functionalities and workflows in your project.
