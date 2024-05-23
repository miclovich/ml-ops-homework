from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

# Read the parquet file
df = pd.read_parquet('yellow_tripdata_2023-01.parquet')
# breakpoint()

# Assuming X is your feature matrix and y is the response variable
X = df.drop(columns=['passenger_count'])

y = df['passenger_count']
model = LinearRegression()

# Fit the model to the data
model.fit(X, y)

# Predict the response variable
y_pred = model.predict(X)

# Calculate the RMSE of the model on the training data
rmse = np.sqrt(mean_squared_error(y, y_pred))
print('Root Mean Squared Error:', rmse)