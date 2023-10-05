#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import scipy.stats as st
# Read dataset into a DataFrame
df = pd.read_csv("po2_data.csv")


# In[2]:


print(df)
print(df.head())


# In[3]:


print(df.describe())


# In[4]:


print(df.isnull().sum())


# # Task1

# In[5]:


x = df[['age', 'sex', 'jitter(%)', 
              'jitter(abs)', 'test_time', 'jitter(rap)', 'jitter(ppq5)',
              'jitter(ddp)', 'shimmer(%)', 'shimmer(abs)', 
              'shimmer(apq3)', 'shimmer(apq5)', 'shimmer(apq11)', 
              'shimmer(dda)', 'nhr', 'hnr', 'rpde', 'dfa', 'ppe']] 

y_mobile_updrs = df['motor_updrs']
y_total_updrs = df['total_updrs']


# In[6]:


print(x)


# # motor_updrs

# In[7]:


# Split dataset into 60% training and 40% test sets 
# Note: other % split can be used.
X_train, X_test, y_train_m, y_test_m = train_test_split(x, y_mobile_updrs, test_size=0.4, random_state=0)


# In[8]:


# Build a linear regression model
model = LinearRegression()


# In[9]:


# Train (fit) the linear regression model using the training set
model.fit(X_train, y_train_m)

# Print the intercept and coefficient learned by the linear regression model
print("Intercept: ", model.intercept_)
print("Coefficient: ", model.coef_)

# Use linear regression to predict the values of (y) in the test set
# based on the values of x in the test set
y_pred = model.predict(X_test)

# Optional: Show the predicted values of (y) next to the actual values of (y)
df_pred = pd.DataFrame({"Actual": y_test_m, "Predicted": y_pred})
print(df_pred)

# Compute standard performance metrics of the linear regression:

# Mean Absolute Error
mae = metrics.mean_absolute_error(y_test_m, y_pred)
# Mean Squared Error
mse = metrics.mean_squared_error(y_test_m, y_pred)
# Root Mean Square Error
rmse =  math.sqrt(metrics.mean_squared_error(y_test_m, y_pred))
# Normalised Root Mean Square Error
y_max = y_mobile_updrs.max()
y_min = y_mobile_updrs.min()
rmse_norm = rmse / (y_max - y_min)
# R-Squared
r_2 = metrics.r2_score(y_test_m, y_pred)
Adj_r2 = 1 - (1-r_2) * (len(y_mobile_updrs)-1)/(len(y_mobile_updrs)-x.shape[1]-1)
 
print("MLP performance:")
print("MAE: ", mae)
print("MSE: ", mse)
print("RMSE: ", rmse)
print("RMSE (Normalised): ", rmse_norm)
print("R^2: ", r_2)
print("Adjusted R^2", Adj_r2)


# Performance metrics and their corresponding values
metric = ['MAE', 'MSE', 'RMSE', 'RMSE (Normalized)', 'R^2', 'Adjusted R^2']
values = [mae, mse, rmse, rmse_norm, r_2, Adj_r2]

# Create a bar graph with values on top
plt.figure(figsize=(8, 6))
bars = plt.bar(metric, values, color='skyblue')

# Display the values on top of the bars
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2 - 0.15, bar.get_height() + 0.2, round(value, 2), ha='center')

plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('MLP Performance Metrics')
plt.ylim(0, max(values) + 10)  # Setting y-axis limit for better visualization
plt.show()


# # total updrs

# In[10]:


# Split dataset into 60% training and 40% test sets 
# Note: other % split can be used.
X_train, X_test, y_train_t, y_test_t = train_test_split(x, y_total_updrs, test_size=0.4, random_state=0)


# In[11]:


# Build a linear regression model
model = LinearRegression()


# In[12]:


# Train (fit) the linear regression model using the training set
model.fit(X_train, y_train_t)

# Print the intercept and coefficient learned by the linear regression model
print("Intercept: ", model.intercept_)
print("Coefficient: ", model.coef_)

# Use linear regression to predict the values of (y) in the test set
# based on the values of x in the test set
y_pred = model.predict(X_test)

# Optional: Show the predicted values of (y) next to the actual values of (y)
df_pred = pd.DataFrame({"Actual": y_test_t, "Predicted": y_pred})
print(df_pred)

# Compute standard performance metrics of the linear regression:

# Mean Absolute Error
mae = metrics.mean_absolute_error(y_test_t, y_pred)
# Mean Squared Error
mse = metrics.mean_squared_error(y_test_t, y_pred)
# Root Mean Square Error
rmse =  math.sqrt(metrics.mean_squared_error(y_test_t, y_pred))
# Normalised Root Mean Square Error
y_max = y_total_updrs.max()
y_min = y_total_updrs.min()
rmse_norm = rmse / (y_max - y_min)

# R-Squared
r_2 = metrics.r2_score(y_test_t, y_pred)
Adj_r2 = 1 - (1-r_2) * (len(y_total_updrs)-1)/(len(y_total_updrs)-x.shape[1]-1)
print("MLP performance:")
print("MAE: ", mae)
print("MSE: ", mse)
print("RMSE: ", rmse)
print("RMSE (Normalised): ", rmse_norm)
print("R^2: ", r_2)
print("Adjusted R^2", Adj_r2)

# Performance metrics and their corresponding values
metric = ['MAE', 'MSE', 'RMSE', 'RMSE (Normalized)', 'R^2', 'Adjusted R^2']
values = [mae, mse, rmse, rmse_norm, r_2, Adj_r2]

# Create a bar graph with values on top
plt.figure(figsize=(8, 6))
bars = plt.bar(metric, values, color='skyblue')

# Display the values on top of the bars
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2 - 0.15, bar.get_height() + 0.2, round(value, 2), ha='center')

plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('MLP Performance Metrics')
plt.ylim(0, max(values) + 10)  # Setting y-axis limit for better visualization
plt.show()


# # Part 2

# # 50-50 motor_updrs

# In[13]:


# Split dataset into 50% training and 40% test sets 
# Note: other % split can be used.
X_train, X_test, y_train_m, y_test_m = train_test_split(x, y_mobile_updrs, test_size=0.5, random_state=0)

# Build a linear regression model
model = LinearRegression()
# Train (fit) the linear regression model using the training set
model.fit(X_train, y_train_m)

# Print the intercept and coefficient learned by the linear regression model
print("Intercept: ", model.intercept_)
print("Coefficient: ", model.coef_)

# Use linear regression to predict the values of (y) in the test set
# based on the values of x in the test set
y_pred = model.predict(X_test)

# Optional: Show the predicted values of (y) next to the actual values of (y)
df_pred = pd.DataFrame({"Actual": y_test_m, "Predicted": y_pred})
print(df_pred)

# Compute standard performance metrics of the linear regression:

# Mean Absolute Error
mae = metrics.mean_absolute_error(y_test_m, y_pred)
# Mean Squared Error
mse = metrics.mean_squared_error(y_test_m, y_pred)
# Root Mean Square Error
rmse =  math.sqrt(metrics.mean_squared_error(y_test_m, y_pred))
# Normalised Root Mean Square Error
y_max = y_mobile_updrs.max()
y_min = y_mobile_updrs.min()
rmse_norm = rmse / (y_max - y_min)

# R-Squared
r_2 = metrics.r2_score(y_test_m, y_pred)
Adj_r2 = 1 - (1-r_2) * (len(y_mobile_updrs)-1)/(len(y_mobile_updrs)-x.shape[1]-1)
print("MLP performance:")
print("MAE: ", mae)
print("MSE: ", mse)
print("RMSE: ", rmse)
print("RMSE (Normalised): ", rmse_norm)
print("R^2: ", r_2)
print("Adjusted R^2", Adj_r2)

# Performance metrics and their corresponding values
metric = ['MAE', 'MSE', 'RMSE', 'RMSE (Normalized)', 'R^2', 'Adjusted R^2']
values = [mae, mse, rmse, rmse_norm, r_2, Adj_r2]

# Create a bar graph with values on top
plt.figure(figsize=(8, 6))
bars = plt.bar(metric, values, color='skyblue')

# Display the values on top of the bars
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2 - 0.15, bar.get_height() + 0.2, round(value, 2), ha='center')

plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('MLP Performance Metrics')
plt.ylim(0, max(values) + 10)  # Setting y-axis limit for better visualization
plt.show()


# # 60-40 motor_updrs

# In[14]:


# Split dataset into 60% training and 40% test sets 
# Note: other % split can be used.
X_train, X_test, y_train_m, y_test_m = train_test_split(x, y_mobile_updrs, test_size=0.4, random_state=0)

# Build a linear regression model
model = LinearRegression()
# Train (fit) the linear regression model using the training set
model.fit(X_train, y_train_m)

# Print the intercept and coefficient learned by the linear regression model
print("Intercept: ", model.intercept_)
print("Coefficient: ", model.coef_)

# Use linear regression to predict the values of (y) in the test set
# based on the values of x in the test set
y_pred = model.predict(X_test)

# Optional: Show the predicted values of (y) next to the actual values of (y)
df_pred = pd.DataFrame({"Actual": y_test_m, "Predicted": y_pred})
print(df_pred)

# Compute standard performance metrics of the linear regression:

# Mean Absolute Error
mae = metrics.mean_absolute_error(y_test_m, y_pred)
# Mean Squared Error
mse = metrics.mean_squared_error(y_test_m, y_pred)
# Root Mean Square Error
rmse =  math.sqrt(metrics.mean_squared_error(y_test_m, y_pred))
# Normalised Root Mean Square Error
y_max = y_mobile_updrs.max()
y_min = y_mobile_updrs.min()
rmse_norm = rmse / (y_max - y_min)

# R-Squared
r_2 = metrics.r2_score(y_test_m, y_pred)
Adj_r2 = 1 - (1-r_2) * (len(y_mobile_updrs)-1)/(len(y_mobile_updrs)-x.shape[1]-1)
print("MLP performance:")
print("MAE: ", mae)
print("MSE: ", mse)
print("RMSE: ", rmse)
print("RMSE (Normalised): ", rmse_norm)
print("R^2: ", r_2)
print("Adjusted R^2", Adj_r2)

# Performance metrics and their corresponding values
metric = ['MAE', 'MSE', 'RMSE', 'RMSE (Normalized)', 'R^2', 'Adjusted R^2']
values = [mae, mse, rmse, rmse_norm, r_2, Adj_r2]

# Create a bar graph with values on top
plt.figure(figsize=(8, 6))
bars = plt.bar(metric, values, color='skyblue')

# Display the values on top of the bars
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2 - 0.15, bar.get_height() + 0.2, round(value, 2), ha='center')

plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('MLP Performance Metrics')
plt.ylim(0, max(values) + 10)  # Setting y-axis limit for better visualization
plt.show()


# # 70-30 motor_updrs

# In[15]:


# Split dataset into 70% training and 30% test sets 
# Note: other % split can be used.
X_train, X_test, y_train_m, y_test_m = train_test_split(x, y_mobile_updrs, test_size=0.3, random_state=0)

# Build a linear regression model
model = LinearRegression()
# Train (fit) the linear regression model using the training set
model.fit(X_train, y_train_m)

# Print the intercept and coefficient learned by the linear regression model
print("Intercept: ", model.intercept_)
print("Coefficient: ", model.coef_)

# Use linear regression to predict the values of (y) in the test set
# based on the values of x in the test set
y_pred = model.predict(X_test)

# Optional: Show the predicted values of (y) next to the actual values of (y)
df_pred = pd.DataFrame({"Actual": y_test_m, "Predicted": y_pred})
print(df_pred)

# Compute standard performance metrics of the linear regression:

# Mean Absolute Error
mae = metrics.mean_absolute_error(y_test_m, y_pred)
# Mean Squared Error
mse = metrics.mean_squared_error(y_test_m, y_pred)
# Root Mean Square Error
rmse =  math.sqrt(metrics.mean_squared_error(y_test_m, y_pred))
# Normalised Root Mean Square Error
y_max = y_mobile_updrs.max()
y_min = y_mobile_updrs.min()
rmse_norm = rmse / (y_max - y_min)

# R-Squared
r_2 = metrics.r2_score(y_test_m, y_pred)
Adj_r2 = 1 - (1-r_2) * (len(y_mobile_updrs)-1)/(len(y_mobile_updrs)-x.shape[1]-1)
print("MLP performance:")
print("MAE: ", mae)
print("MSE: ", mse)
print("RMSE: ", rmse)
print("RMSE (Normalised): ", rmse_norm)
print("R^2: ", r_2)
print("Adjusted R^2", Adj_r2)

# Performance metrics and their corresponding values
metric = ['MAE', 'MSE', 'RMSE', 'RMSE (Normalized)', 'R^2', 'Adjusted R^2']
values = [mae, mse, rmse, rmse_norm, r_2, Adj_r2]

# Create a bar graph with values on top
plt.figure(figsize=(8, 6))
bars = plt.bar(metric, values, color='skyblue')

# Display the values on top of the bars
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2 - 0.15, bar.get_height() + 0.2, round(value, 2), ha='center')

plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('MLP Performance Metrics')
plt.ylim(0, max(values) + 10)  # Setting y-axis limit for better visualization
plt.show()


# # 80-20 motor_updrs

# In[16]:


# Split dataset into 80% training and 20% test sets 
# Note: other % split can be used.
X_train, X_test, y_train_m, y_test_m = train_test_split(x, y_mobile_updrs, test_size=0.2, random_state=0)

# Build a linear regression model
model = LinearRegression()
# Train (fit) the linear regression model using the training set
model.fit(X_train, y_train_m)

# Print the intercept and coefficient learned by the linear regression model
print("Intercept: ", model.intercept_)
print("Coefficient: ", model.coef_)

# Use linear regression to predict the values of (y) in the test set
# based on the values of x in the test set
y_pred = model.predict(X_test)

# Optional: Show the predicted values of (y) next to the actual values of (y)
df_pred = pd.DataFrame({"Actual": y_test_m, "Predicted": y_pred})
print(df_pred)

# Compute standard performance metrics of the linear regression:

# Mean Absolute Error
mae = metrics.mean_absolute_error(y_test_m, y_pred)
# Mean Squared Error
mse = metrics.mean_squared_error(y_test_m, y_pred)
# Root Mean Square Error
rmse =  math.sqrt(metrics.mean_squared_error(y_test_m, y_pred))
# Normalised Root Mean Square Error
y_max = y_mobile_updrs.max()
y_min = y_mobile_updrs.min()
rmse_norm = rmse / (y_max - y_min)

# R-Squared
r_2 = metrics.r2_score(y_test_m, y_pred)
Adj_r2 = 1 - (1-r_2) * (len(y_mobile_updrs)-1)/(len(y_mobile_updrs)-x.shape[1]-1)
print("MLP performance:")
print("MAE: ", mae)
print("MSE: ", mse)
print("RMSE: ", rmse)
print("RMSE (Normalised): ", rmse_norm)
print("R^2: ", r_2)
print("Adjusted R^2", Adj_r2)

# Performance metrics and their corresponding values
metric = ['MAE', 'MSE', 'RMSE', 'RMSE (Normalized)', 'R^2', 'Adjusted R^2']
values = [mae, mse, rmse, rmse_norm, r_2, Adj_r2]

# Create a bar graph with values on top
plt.figure(figsize=(8, 6))
bars = plt.bar(metric, values, color='skyblue')

# Display the values on top of the bars
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2 - 0.15, bar.get_height() + 0.2, round(value, 2), ha='center')

plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('MLP Performance Metrics')
plt.ylim(0, max(values) + 10)  # Setting y-axis limit for better visualization
plt.show()


# # 50-50 total_updrs

# In[17]:


# Split dataset into 50% training and 50% test sets 
# Note: other % split can be used.
X_train, X_test, y_train_t, y_test_t = train_test_split(x, y_total_updrs, test_size=0.5, random_state=0)

# Build a linear regression model
model = LinearRegression()
# Train (fit) the linear regression model using the training set
model.fit(X_train, y_train_t)

# Print the intercept and coefficient learned by the linear regression model
print("Intercept: ", model.intercept_)
print("Coefficient: ", model.coef_)

# Use linear regression to predict the values of (y) in the test set
# based on the values of x in the test set
y_pred = model.predict(X_test)

# Optional: Show the predicted values of (y) next to the actual values of (y)
df_pred = pd.DataFrame({"Actual": y_test_t, "Predicted": y_pred})
print(df_pred)

# Compute standard performance metrics of the linear regression:

# Mean Absolute Error
mae = metrics.mean_absolute_error(y_test_t, y_pred)
# Mean Squared Error
mse = metrics.mean_squared_error(y_test_t, y_pred)
# Root Mean Square Error
rmse =  math.sqrt(metrics.mean_squared_error(y_test_t, y_pred))
# Normalised Root Mean Square Error
y_max = y_total_updrs.max()
y_min = y_total_updrs.min()
rmse_norm = rmse / (y_max - y_min)

# R-Squared
r_2 = metrics.r2_score(y_test_t, y_pred)
Adj_r2 = 1 - (1-r_2) * (len(y_total_updrs)-1)/(len(y_total_updrs)-x.shape[1]-1)
print("MLP performance:")
print("MAE: ", mae)
print("MSE: ", mse)
print("RMSE: ", rmse)
print("RMSE (Normalised): ", rmse_norm)
print("R^2: ", r_2)
print("Adjusted R^2", Adj_r2)

# Performance metrics and their corresponding values
metric = ['MAE', 'MSE', 'RMSE', 'RMSE (Normalized)', 'R^2', 'Adjusted R^2']
values = [mae, mse, rmse, rmse_norm, r_2, Adj_r2]

# Create a bar graph with values on top
plt.figure(figsize=(8, 6))
bars = plt.bar(metric, values, color='skyblue')

# Display the values on top of the bars
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2 - 0.15, bar.get_height() + 0.2, round(value, 2), ha='center')

plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('MLP Performance Metrics')
plt.ylim(0, max(values) + 10)  # Setting y-axis limit for better visualization
plt.show()


# # 60-40 total_updrs

# In[18]:


# Split dataset into 60% training and 40% test sets 
# Note: other % split can be used.
X_train, X_test, y_train_t, y_test_t = train_test_split(x, y_total_updrs, test_size=0.4, random_state=0)

# Build a linear regression model
model = LinearRegression()
# Train (fit) the linear regression model using the training set
model.fit(X_train, y_train_t)

# Print the intercept and coefficient learned by the linear regression model
print("Intercept: ", model.intercept_)
print("Coefficient: ", model.coef_)

# Use linear regression to predict the values of (y) in the test set
# based on the values of x in the test set
y_pred = model.predict(X_test)

# Optional: Show the predicted values of (y) next to the actual values of (y)
df_pred = pd.DataFrame({"Actual": y_test_t, "Predicted": y_pred})
print(df_pred)

# Compute standard performance metrics of the linear regression:

# Mean Absolute Error
mae = metrics.mean_absolute_error(y_test_t, y_pred)
# Mean Squared Error
mse = metrics.mean_squared_error(y_test_t, y_pred)
# Root Mean Square Error
rmse =  math.sqrt(metrics.mean_squared_error(y_test_t, y_pred))
# Normalised Root Mean Square Error
y_max = y_total_updrs.max()
y_min = y_total_updrs.min()
rmse_norm = rmse / (y_max - y_min)

# R-Squared
r_2 = metrics.r2_score(y_test_t, y_pred)
Adj_r2 = 1 - (1-r_2) * (len(y_total_updrs)-1)/(len(y_total_updrs)-x.shape[1]-1)
print("MLP performance:")
print("MAE: ", mae)
print("MSE: ", mse)
print("RMSE: ", rmse)
print("RMSE (Normalised): ", rmse_norm)
print("R^2: ", r_2)
print("Adjusted R^2", Adj_r2)

# Performance metrics and their corresponding values
metric = ['MAE', 'MSE', 'RMSE', 'RMSE (Normalized)', 'R^2', 'Adjusted R^2']
values = [mae, mse, rmse, rmse_norm, r_2, Adj_r2]

# Create a bar graph with values on top
plt.figure(figsize=(8, 6))
bars = plt.bar(metric, values, color='skyblue')

# Display the values on top of the bars
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2 - 0.15, bar.get_height() + 0.2, round(value, 2), ha='center')

plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('MLP Performance Metrics')
plt.ylim(0, max(values) + 10)  # Setting y-axis limit for better visualization
plt.show()


# # 70-30 total_updrs

# In[19]:


# Split dataset into 70% training and 30% test sets 
# Note: other % split can be used.
X_train, X_test, y_train_t, y_test_t = train_test_split(x, y_total_updrs, test_size=0.3, random_state=0)

# Build a linear regression model
model = LinearRegression()
# Train (fit) the linear regression model using the training set
model.fit(X_train, y_train_t)

# Print the intercept and coefficient learned by the linear regression model
print("Intercept: ", model.intercept_)
print("Coefficient: ", model.coef_)

# Use linear regression to predict the values of (y) in the test set
# based on the values of x in the test set
y_pred = model.predict(X_test)

# Optional: Show the predicted values of (y) next to the actual values of (y)
df_pred = pd.DataFrame({"Actual": y_test_t, "Predicted": y_pred})
print(df_pred)

# Compute standard performance metrics of the linear regression:

# Mean Absolute Error
mae = metrics.mean_absolute_error(y_test_t, y_pred)
# Mean Squared Error
mse = metrics.mean_squared_error(y_test_t, y_pred)
# Root Mean Square Error
rmse =  math.sqrt(metrics.mean_squared_error(y_test_t, y_pred))
# Normalised Root Mean Square Error
y_max = y_total_updrs.max()
y_min = y_total_updrs.min()
rmse_norm = rmse / (y_max - y_min)

# R-Squared
r_2 = metrics.r2_score(y_test_t, y_pred)
Adj_r2 = 1 - (1-r_2) * (len(y_total_updrs)-1)/(len(y_total_updrs)-x.shape[1]-1)
print("MLP performance:")
print("MAE: ", mae)
print("MSE: ", mse)
print("RMSE: ", rmse)
print("RMSE (Normalised): ", rmse_norm)
print("R^2: ", r_2)
print("Adjusted R^2", Adj_r2)

# Performance metrics and their corresponding values
metric = ['MAE', 'MSE', 'RMSE', 'RMSE (Normalized)', 'R^2', 'Adjusted R^2']
values = [mae, mse, rmse, rmse_norm, r_2, Adj_r2]

# Create a bar graph with values on top
plt.figure(figsize=(8, 6))
bars = plt.bar(metric, values, color='skyblue')

# Display the values on top of the bars
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2 - 0.15, bar.get_height() + 0.2, round(value, 2), ha='center')

plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('MLP Performance Metrics')
plt.ylim(0, max(values) + 10)  # Setting y-axis limit for better visualization
plt.show()


# # 80-20 total_updrs

# In[20]:


# Split dataset into 80% training and 20% test sets 
# Note: other % split can be used.
X_train, X_test, y_train_t, y_test_t = train_test_split(x, y_total_updrs, test_size=0.2, random_state=0)

# Build a linear regression model
model = LinearRegression()
# Train (fit) the linear regression model using the training set
model.fit(X_train, y_train_t)

# Print the intercept and coefficient learned by the linear regression model
print("Intercept: ", model.intercept_)
print("Coefficient: ", model.coef_)

# Use linear regression to predict the values of (y) in the test set
# based on the values of x in the test set
y_pred = model.predict(X_test)

# Optional: Show the predicted values of (y) next to the actual values of (y)
df_pred = pd.DataFrame({"Actual": y_test_t, "Predicted": y_pred})
print(df_pred)

# Compute standard performance metrics of the linear regression:

# Mean Absolute Error
mae = metrics.mean_absolute_error(y_test_t, y_pred)
# Mean Squared Error
mse = metrics.mean_squared_error(y_test_t, y_pred)
# Root Mean Square Error
rmse =  math.sqrt(metrics.mean_squared_error(y_test_t, y_pred))
# Normalised Root Mean Square Error
y_max = y_total_updrs.max()
y_min = y_total_updrs.min()
rmse_norm = rmse / (y_max - y_min)

# R-Squared
r_2 = metrics.r2_score(y_test_t, y_pred)
Adj_r2 = 1 - (1-r_2) * (len(y_total_updrs)-1)/(len(y_total_updrs)-x.shape[1]-1)
print("MLP performance:")
print("MAE: ", mae)
print("MSE: ", mse)
print("RMSE: ", rmse)
print("RMSE (Normalised): ", rmse_norm)
print("R^2: ", r_2)
print("Adjusted R^2", Adj_r2)

# Performance metrics and their corresponding values
metric = ['MAE', 'MSE', 'RMSE', 'RMSE (Normalized)', 'R^2', 'Adjusted R^2']
values = [mae, mse, rmse, rmse_norm, r_2, Adj_r2]

# Create a bar graph with values on top
plt.figure(figsize=(8, 6))
bars = plt.bar(metric, values, color='skyblue')

# Display the values on top of the bars
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2 - 0.15, bar.get_height() + 0.2, round(value, 2), ha='center')

plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('MLP Performance Metrics')
plt.ylim(0, max(values) + 10)  # Setting y-axis limit for better visualization
plt.show()


# # part 3

# In[21]:


df = pd.read_csv("po2_data.csv")
x = df[['age', 'sex', 'jitter(%)', 
              'jitter(abs)', 'test_time', 'jitter(rap)', 'jitter(ppq5)',
              'jitter(ddp)', 'shimmer(%)', 'shimmer(abs)', 
              'shimmer(apq3)', 'shimmer(apq5)', 'shimmer(apq11)', 
              'shimmer(dda)', 'nhr', 'hnr', 'rpde', 'dfa', 'ppe']] 

y_mobile_updrs = df['motor_updrs']
y_total_updrs = df['total_updrs']


# In[22]:


column_names = list(x.columns)
print(column_names)


# In[23]:


for i in range(0, 19):
    column_name = column_names[i]
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column_name] < lower_bound) | (df[column_name] > upper_bound)]
#     df = df[(df[column_name] < lower_bound) | (df[column_name] > upper_bound)]
    fig1, ax1 = plt.subplots()
    fig1.tight_layout()
    ax1.scatter(x=df[column_name], y=df['motor_updrs'])
    ax1.set_xlabel(column_name)
    ax1.set_ylabel("motor_updrs")
    
    # Highlight outliers on the plot
    ax1.scatter(x=outliers[column_name], y=outliers['motor_updrs'], color='red', label='Outliers')
    ax1.legend()
    
    plt.show()


# In[24]:


for i in range(0, 19):
    column_name = column_names[i]
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column_name] < lower_bound) | (df[column_name] > upper_bound)]
#     df = df[(df[column_name] < lower_bound) | (df[column_name] > upper_bound)]
    fig1, ax1 = plt.subplots()
    fig1.tight_layout()
    ax1.scatter(x=df[column_name], y=df['total_updrs'])
    ax1.set_xlabel(column_name)
    ax1.set_ylabel("total_updrs")
    
    # Highlight outliers on the plot
    ax1.scatter(x=outliers[column_name], y=outliers['total_updrs'], color='red', label='Outliers')
    ax1.legend()
    
    plt.show()


# # Box-plot on independent variables

# In[25]:


for i in range(0,19):
    
    sns.set_theme(style="whitegrid")  # optional
    ax = sns.boxplot(data=df[column_names[i]], orient="h", palette="Set2", whis=1.5)
    ax.set_xlabel(column_names[i])
    plt.show()


# In[26]:


# Apply non-linear transformation
df["hnr_lt"] = df["hnr"].apply(np.log)
df["rpde_lt"] = df["rpde"].apply(np.log)
df["ppe_lt"] = df["ppe"].apply(np.log)


# In[27]:


a= df.columns
print(a)


# In[28]:


# Visualise the effect of the transformation
plt.figure(figsize=(20,10))

plt.subplot(1,2,1)
plt.scatter(df["hnr"], df["motor_updrs"], color="green")
plt.title("Original hnr")
plt.xlabel("hnr")
plt.ylabel("motor_updrs")
plt.plot([30,15],[0,40])

plt.subplot(1,2,2)
plt.scatter(df["hnr_lt"], df["motor_updrs"], color="red")
plt.title("Log Transformed hnr")
plt.xlabel("hnr_lt")
plt.ylabel("motor_updrs")
plt.plot([3.5,2.5],[0,40])

plt.show()

# Visualise the effect of the transformation
plt.figure(figsize=(20,10))

plt.subplot(1,2,1)
plt.scatter(df["rpde"], df["motor_updrs"], color="green")
plt.title("Original rpde")
plt.xlabel("rpde")
plt.ylabel("motor_updrs")
plt.plot([0.2,0.8],[0,40])

plt.subplot(1,2,2)
plt.scatter(df["rpde_lt"], df["motor_updrs"], color="red")
plt.title("Log Transformed rpde")
plt.xlabel("rpde_lt")
plt.ylabel("motor_updrs")
plt.plot([-1.5,-.25],[0,40])

plt.show()

# Visualise the effect of the transformation
plt.figure(figsize=(20,10))

plt.subplot(1,2,1)
plt.scatter(df["ppe"], df["motor_updrs"], color="green")
plt.title("Original ppe")
plt.xlabel("ppe")
plt.ylabel("motor_updrs")
plt.plot([0,0.4],[0,40])

plt.subplot(1,2,2)
plt.scatter(df["ppe_lt"], df["motor_updrs"], color="red")
plt.title("Log Transformed ppe")
plt.xlabel("ppe_lt")
plt.ylabel("motor_updrs")
plt.plot([-2.5,-1],[0,40])

plt.show()


# In[29]:


print(a)


# In[ ]:





# In[30]:


# Visualise the effect of the transformation
plt.figure(figsize=(20,10))

plt.subplot(1,2,1)
plt.scatter(df["hnr"], df["total_updrs"], color="green")
plt.title("Original hnr")
plt.xlabel("hnr")
plt.ylabel("total_updrs")
plt.plot([30,15],[0,60])

plt.subplot(1,2,2)
plt.scatter(df["hnr_lt"], df["total_updrs"], color="red")
plt.title("Log Transformed hnr")
plt.xlabel("hnr_lt")
plt.ylabel("total_updrs")
plt.plot([3.5,2.5],[0,60])

plt.show()

# Visualise the effect of the transformation
plt.figure(figsize=(20,10))

plt.subplot(1,2,1)
plt.scatter(df["rpde"], df["total_updrs"], color="green")
plt.title("Original rpde")
plt.xlabel("rpde")
plt.ylabel("total_updrs")
plt.plot([0.2,0.8],[0,60])

plt.subplot(1,2,2)
plt.scatter(df["rpde_lt"], df["total_updrs"], color="red")
plt.title("Log Transformed rpde")
plt.xlabel("rpde_lt")
plt.ylabel("total_updrs")
plt.plot([-1.5,-.25],[0,60])

plt.show()

# Visualise the effect of the transformation
plt.figure(figsize=(20,10))

plt.subplot(1,2,1)
plt.scatter(df["ppe"], df["total_updrs"], color="green")
plt.title("Original ppe")
plt.xlabel("ppe")
plt.ylabel("total_updrs")
plt.plot([0,0.4],[0,60])

plt.subplot(1,2,2)
plt.scatter(df["ppe_lt"], df["total_updrs"], color="red")
plt.title("Log Transformed ppe")
plt.xlabel("ppe_lt")
plt.ylabel("total_updrs")
plt.plot([-2.5,-1],[0,60])

plt.show()


# In[31]:


#Drop the original LSTAT variable
df = df.drop("hnr", axis=1)
df = df.drop("rpde", axis=1)
df = df.drop("ppe", axis=1)


# In[32]:


a= df.columns
print(a)


# In[33]:


x = df[['age', 'sex', 'jitter(%)', 
              'jitter(abs)', 'test_time', 'jitter(rap)', 'jitter(ppq5)',
              'jitter(ddp)', 'shimmer(%)', 'shimmer(abs)', 
              'shimmer(apq3)', 'shimmer(apq5)', 'shimmer(apq11)', 
              'shimmer(dda)', 'nhr', 'hnr_lt', 'rpde_lt', 'dfa', 'ppe_lt']] 

y_mobile_updrs = df['motor_updrs']
y_total_updrs = df['total_updrs']


# In[34]:


print(x)


# In[35]:


# Split dataset into 60% training and 40% test sets 
# Note: other % split can be used.
X_train, X_test, y_train_m, y_test_m = train_test_split(x, y_mobile_updrs, test_size=0.4, random_state=0)

# Build a linear regression model
model = LinearRegression()
# Train (fit) the linear regression model using the training set
model.fit(X_train, y_train_m)

# Print the intercept and coefficient learned by the linear regression model
print("Intercept: ", model.intercept_)
print("Coefficient: ", model.coef_)

# Use linear regression to predict the values of (y) in the test set
# based on the values of x in the test set
y_pred = model.predict(X_test)

# Optional: Show the predicted values of (y) next to the actual values of (y)
df_pred = pd.DataFrame({"Actual": y_test_m, "Predicted": y_pred})
print(df_pred)

# Compute standard performance metrics of the linear regression:

# Mean Absolute Error
mae = metrics.mean_absolute_error(y_test_m, y_pred)
# Mean Squared Error
mse = metrics.mean_squared_error(y_test_m, y_pred)
# Root Mean Square Error
rmse =  math.sqrt(metrics.mean_squared_error(y_test_m, y_pred))
# Normalised Root Mean Square Error
y_max = y_mobile_updrs.max()
y_min = y_mobile_updrs.min()
rmse_norm = rmse / (y_max - y_min)

# R-Squared
r_2 = metrics.r2_score(y_test_m, y_pred)
Adj_r2 = 1 - (1-r_2) * (len(y_mobile_updrs)-1)/(len(y_mobile_updrs)-x.shape[1]-1)
print("MLP performance:")
print("MAE: ", mae)
print("MSE: ", mse)
print("RMSE: ", rmse)
print("RMSE (Normalised): ", rmse_norm)
print("R^2: ", r_2)
print("Adjusted R^2", Adj_r2)

# Performance metrics and their corresponding values
metric = ['MAE', 'MSE', 'RMSE', 'RMSE (Normalized)', 'R^2', 'Adjusted R^2']
values = [mae, mse, rmse, rmse_norm, r_2, Adj_r2]

# Create a bar graph with values on top
plt.figure(figsize=(8, 6))
bars = plt.bar(metric, values, color='skyblue')

# Display the values on top of the bars
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2 - 0.15, bar.get_height() + 0.2, round(value, 2), ha='center')

plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('MLP Performance Metrics')
plt.ylim(0, max(values) + 10)  # Setting y-axis limit for better visualization
plt.show()


# In[36]:


# Split dataset into 60% training and 40% test sets 
# Note: other % split can be used.
X_train, X_test, y_train_t, y_test_t = train_test_split(x, y_total_updrs, test_size=0.4, random_state=0)

# Build a linear regression model
model = LinearRegression()
# Train (fit) the linear regression model using the training set
model.fit(X_train, y_train_t)

# Print the intercept and coefficient learned by the linear regression model
print("Intercept: ", model.intercept_)
print("Coefficient: ", model.coef_)

# Use linear regression to predict the values of (y) in the test set
# based on the values of x in the test set
y_pred = model.predict(X_test)

# Optional: Show the predicted values of (y) next to the actual values of (y)
df_pred = pd.DataFrame({"Actual": y_test_t, "Predicted": y_pred})
print(df_pred)

# Compute standard performance metrics of the linear regression:

# Mean Absolute Error
mae = metrics.mean_absolute_error(y_test_t, y_pred)
# Mean Squared Error
mse = metrics.mean_squared_error(y_test_t, y_pred)
# Root Mean Square Error
rmse =  math.sqrt(metrics.mean_squared_error(y_test_t, y_pred))
# Normalised Root Mean Square Error
y_max = y_total_updrs.max()
y_min = y_total_updrs.min()
rmse_norm = rmse / (y_max - y_min)

# R-Squared
r_2 = metrics.r2_score(y_test_t, y_pred)
Adj_r2 = 1 - (1-r_2) * (len(y_total_updrs)-1)/(len(y_total_updrs)-x.shape[1]-1)
print("MLP performance:")
print("MAE: ", mae)
print("MSE: ", mse)
print("RMSE: ", rmse)
print("RMSE (Normalised): ", rmse_norm)
print("R^2: ", r_2)
print("Adjusted R^2", Adj_r2)

# Performance metrics and their corresponding values
metric = ['MAE', 'MSE', 'RMSE', 'RMSE (Normalized)', 'R^2', 'Adjusted R^2']
values = [mae, mse, rmse, rmse_norm, r_2, Adj_r2]

# Create a bar graph with values on top
plt.figure(figsize=(8, 6))
bars = plt.bar(metric, values, color='skyblue')

# Display the values on top of the bars
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2 - 0.15, bar.get_height() + 0.2, round(value, 2), ha='center')

plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('MLP Performance Metrics')
plt.ylim(0, max(values) + 10)  # Setting y-axis limit for better visualization
plt.show()


# # colinearity analysis

# In[37]:


# Plot correlation matrix
corr = x.corr()

# Plot the pairwise correlation as heatmap
plt.figure(figsize=(19,19))
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=False,
    annot=True
)

# customise the labels
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)

plt.show()


# In[38]:


# Drop one or more of the correlated variables. Keep only one.
x = x.drop(["jitter(rap)","jitter(%)","jitter(ppq5)","shimmer(abs)","shimmer(apq3)","shimmer(%)","shimmer(apq5)"], axis=1)
print(x.info())


# In[39]:


# Plot correlation matrix
corr = x.corr()

# Plot the pairwise correlation as heatmap
plt.figure(figsize=(12,12))
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=False,
    annot=True
)

# customise the labels
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)

plt.show()


# In[40]:


# Split dataset into 60% training and 40% test sets 
# Note: other % split can be used.
X_train, X_test, y_train_m, y_test_m = train_test_split(x, y_mobile_updrs, test_size=0.4, random_state=0)

# Build a linear regression model
model = LinearRegression()
# Train (fit) the linear regression model using the training set
model.fit(X_train, y_train_m)

# Print the intercept and coefficient learned by the linear regression model
print("Intercept: ", model.intercept_)
print("Coefficient: ", model.coef_)

# Use linear regression to predict the values of (y) in the test set
# based on the values of x in the test set
y_pred = model.predict(X_test)

# Optional: Show the predicted values of (y) next to the actual values of (y)
df_pred = pd.DataFrame({"Actual": y_test_m, "Predicted": y_pred})
print(df_pred)

# Compute standard performance metrics of the linear regression:

# Mean Absolute Error
mae = metrics.mean_absolute_error(y_test_m, y_pred)
# Mean Squared Error
mse = metrics.mean_squared_error(y_test_m, y_pred)
# Root Mean Square Error
rmse =  math.sqrt(metrics.mean_squared_error(y_test_m, y_pred))
# Normalised Root Mean Square Error
y_max = y_mobile_updrs.max()
y_min = y_mobile_updrs.min()
rmse_norm = rmse / (y_max - y_min)

# R-Squared
r_2 = metrics.r2_score(y_test_m, y_pred)
Adj_r2 = 1 - (1-r_2) * (len(y_mobile_updrs)-1)/(len(y_mobile_updrs)-x.shape[1]-1)
print("MLP performance:")
print("MAE: ", mae)
print("MSE: ", mse)
print("RMSE: ", rmse)
print("RMSE (Normalised): ", rmse_norm)
print("R^2: ", r_2)
print("Adjusted R^2", Adj_r2)

# Performance metrics and their corresponding values
metric = ['MAE', 'MSE', 'RMSE', 'RMSE (Normalized)', 'R^2', 'Adjusted R^2']
values = [mae, mse, rmse, rmse_norm, r_2, Adj_r2]

# Create a bar graph with values on top
plt.figure(figsize=(8, 6))
bars = plt.bar(metric, values, color='skyblue')

# Display the values on top of the bars
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2 - 0.15, bar.get_height() + 0.2, round(value, 2), ha='center')

plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('MLP Performance Metrics')
plt.ylim(0, max(values) + 10)  # Setting y-axis limit for better visualization
plt.show()


# In[41]:


# Split dataset into 60% training and 40% test sets 
# Note: other % split can be used.
X_train, X_test, y_train_t, y_test_t = train_test_split(x, y_total_updrs, test_size=0.4, random_state=0)

# Build a linear regression model
model = LinearRegression()
# Train (fit) the linear regression model using the training set
model.fit(X_train, y_train_t)

# Print the intercept and coefficient learned by the linear regression model
print("Intercept: ", model.intercept_)
print("Coefficient: ", model.coef_)

# Use linear regression to predict the values of (y) in the test set
# based on the values of x in the test set
y_pred = model.predict(X_test)

# Optional: Show the predicted values of (y) next to the actual values of (y)
df_pred = pd.DataFrame({"Actual": y_test_t, "Predicted": y_pred})
print(df_pred)

# Compute standard performance metrics of the linear regression:

# Mean Absolute Error
mae = metrics.mean_absolute_error(y_test_t, y_pred)
# Mean Squared Error
mse = metrics.mean_squared_error(y_test_t, y_pred)
# Root Mean Square Error
rmse =  math.sqrt(metrics.mean_squared_error(y_test_t, y_pred))
# Normalised Root Mean Square Error
y_max = y_total_updrs.max()
y_min = y_total_updrs.min()
rmse_norm = rmse / (y_max - y_min)

# R-Squared
r_2 = metrics.r2_score(y_test_t, y_pred)
Adj_r2 = 1 - (1-r_2) * (len(y_total_updrs)-1)/(len(y_total_updrs)-x.shape[1]-1)
print("MLP performance:")
print("MAE: ", mae)
print("MSE: ", mse)
print("RMSE: ", rmse)
print("RMSE (Normalised): ", rmse_norm)
print("R^2: ", r_2)
print("Adjusted R^2", Adj_r2)

# Performance metrics and their corresponding values
metric = ['MAE', 'MSE', 'RMSE', 'RMSE (Normalized)', 'R^2', 'Adjusted R^2']
values = [mae, mse, rmse, rmse_norm, r_2, Adj_r2]

# Create a bar graph with values on top
plt.figure(figsize=(8, 6))
bars = plt.bar(metric, values, color='skyblue')

# Display the values on top of the bars
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2 - 0.15, bar.get_height() + 0.2, round(value, 2), ha='center')

plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('MLP Performance Metrics')
plt.ylim(0, max(values) + 10)  # Setting y-axis limit for better visualization
plt.show()


# # Part 4

# # standard scaling

# In[42]:


df = pd.read_csv("po2_data.csv")
x = df[['age', 'sex', 'jitter(%)', 
              'jitter(abs)', 'test_time', 'jitter(rap)', 'jitter(ppq5)',
              'jitter(ddp)', 'shimmer(%)', 'shimmer(abs)', 
              'shimmer(apq3)', 'shimmer(apq5)', 'shimmer(apq11)', 
              'shimmer(dda)', 'nhr', 'hnr', 'rpde', 'dfa', 'ppe']] 

y_mobile_updrs = df['motor_updrs']
y_total_updrs = df['total_updrs']


# In[43]:


from sklearn.preprocessing import StandardScaler


# In[44]:


print(x)
scaler = StandardScaler()
# Apply z-score standardisation to all explanatory variables
std_x = scaler.fit_transform(x.values)


# In[45]:


std_x_df = pd.DataFrame(std_x, index=x.index, columns=x.columns)
print(std_x_df)


# In[ ]:





# In[46]:


# Split dataset into 60% training and 40% test sets 
# Note: other % split can be used.
X_train, X_test, y_train_m, y_test_m = train_test_split(std_x_df, y_mobile_updrs, test_size=0.4, random_state=0)

# Build a linear regression model
model = LinearRegression()
# Train (fit) the linear regression model using the training set
model.fit(X_train, y_train_m)

# Print the intercept and coefficient learned by the linear regression model
print("Intercept: ", model.intercept_)
print("Coefficient: ", model.coef_)

# Use linear regression to predict the values of (y) in the test set
# based on the values of x in the test set
y_pred = model.predict(X_test)

# Optional: Show the predicted values of (y) next to the actual values of (y)
df_pred = pd.DataFrame({"Actual": y_test_m, "Predicted": y_pred})
print(df_pred)

# Compute standard performance metrics of the linear regression:

# Mean Absolute Error
mae = metrics.mean_absolute_error(y_test_m, y_pred)
# Mean Squared Error
mse = metrics.mean_squared_error(y_test_m, y_pred)
# Root Mean Square Error
rmse =  math.sqrt(metrics.mean_squared_error(y_test_m, y_pred))
# Normalised Root Mean Square Error
y_max = y_mobile_updrs.max()
y_min = y_mobile_updrs.min()
rmse_norm = rmse / (y_max - y_min)

# R-Squared
r_2 = metrics.r2_score(y_test_m, y_pred)
Adj_r2 = 1 - (1-r_2) * (len(y_mobile_updrs)-1)/(len(y_mobile_updrs)-x.shape[1]-1)
print("MLP performance:")
print("MAE: ", mae)
print("MSE: ", mse)
print("RMSE: ", rmse)
print("RMSE (Normalised): ", rmse_norm)
print("R^2: ", r_2)
print("Adjusted R^2", Adj_r2)

# Performance metrics and their corresponding values
metric = ['MAE', 'MSE', 'RMSE', 'RMSE (Normalized)', 'R^2', 'Adjusted R^2']
values = [mae, mse, rmse, rmse_norm, r_2, Adj_r2]

# Create a bar graph with values on top
plt.figure(figsize=(8, 6))
bars = plt.bar(metric, values, color='skyblue')

# Display the values on top of the bars
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2 - 0.15, bar.get_height() + 0.2, round(value, 2), ha='center')

plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('MLP Performance Metrics')
plt.ylim(0, max(values) + 10)  # Setting y-axis limit for better visualization
plt.show()


# In[47]:


# Split dataset into 60% training and 40% test sets 
# Note: other % split can be used.
X_train, X_test, y_train_t, y_test_t = train_test_split(std_x_df, y_total_updrs, test_size=0.4, random_state=0)

# Build a linear regression model
model = LinearRegression()
# Train (fit) the linear regression model using the training set
model.fit(X_train, y_train_t)

# Print the intercept and coefficient learned by the linear regression model
print("Intercept: ", model.intercept_)
print("Coefficient: ", model.coef_)

# Use linear regression to predict the values of (y) in the test set
# based on the values of x in the test set
y_pred = model.predict(X_test)

# Optional: Show the predicted values of (y) next to the actual values of (y)
df_pred = pd.DataFrame({"Actual": y_test_t, "Predicted": y_pred})
print(df_pred)

# Compute standard performance metrics of the linear regression:

# Mean Absolute Error
mae = metrics.mean_absolute_error(y_test_t, y_pred)
# Mean Squared Error
mse = metrics.mean_squared_error(y_test_t, y_pred)
# Root Mean Square Error
rmse =  math.sqrt(metrics.mean_squared_error(y_test_t, y_pred))
# Normalised Root Mean Square Error
y_max = y_total_updrs.max()
y_min = y_total_updrs.min()
rmse_norm = rmse / (y_max - y_min)

# R-Squared
r_2 = metrics.r2_score(y_test_t, y_pred)
Adj_r2 = 1 - (1-r_2) * (len(y_total_updrs)-1)/(len(y_total_updrs)-x.shape[1]-1)
print("MLP performance:")
print("MAE: ", mae)
print("MSE: ", mse)
print("RMSE: ", rmse)
print("RMSE (Normalised): ", rmse_norm)
print("R^2: ", r_2)
print("Adjusted R^2", Adj_r2)

# Performance metrics and their corresponding values
metric = ['MAE', 'MSE', 'RMSE', 'RMSE (Normalized)', 'R^2', 'Adjusted R^2']
values = [mae, mse, rmse, rmse_norm, r_2, Adj_r2]

# Create a bar graph with values on top
plt.figure(figsize=(8, 6))
bars = plt.bar(metric, values, color='skyblue')

# Display the values on top of the bars
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2 - 0.15, bar.get_height() + 0.2, round(value, 2), ha='center')

plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('MLP Performance Metrics')
plt.ylim(0, max(values) + 10)  # Setting y-axis limit for better visualization
plt.show()


# # gaussian transform

# In[48]:


df = pd.read_csv("po2_data.csv")
x = df[['age', 'sex', 'jitter(%)', 
              'jitter(abs)', 'test_time', 'jitter(rap)', 'jitter(ppq5)',
              'jitter(ddp)', 'shimmer(%)', 'shimmer(abs)', 
              'shimmer(apq3)', 'shimmer(apq5)', 'shimmer(apq11)', 
              'shimmer(dda)', 'nhr', 'hnr', 'rpde', 'dfa', 'ppe']] 

y_mobile_updrs = df['motor_updrs']
y_total_updrs = df['total_updrs']


from sklearn.preprocessing import PowerTransformer
# Create a Yeo-Johnson transformer
scaler = PowerTransformer()

# Apply the transformer to make all explanatory variables more Gaussian-looking
std_x = scaler.fit_transform(x.values)

# Restore column names of explanatory variables
std_x_df = pd.DataFrame(std_x, index=x.index, columns=x.columns)

# Split dataset into 60% training and 40% test sets 
# Note: other % split can be used.
X_train, X_test, y_train_m, y_test_m = train_test_split(std_x_df, y_mobile_updrs, test_size=0.4, random_state=0)

# Build a linear regression model
model = LinearRegression()
# Train (fit) the linear regression model using the training set
model.fit(X_train, y_train_m)

# Print the intercept and coefficient learned by the linear regression model
print("Intercept: ", model.intercept_)
print("Coefficient: ", model.coef_)

# Use linear regression to predict the values of (y) in the test set
# based on the values of x in the test set
y_pred = model.predict(X_test)

# Optional: Show the predicted values of (y) next to the actual values of (y)
df_pred = pd.DataFrame({"Actual": y_test_m, "Predicted": y_pred})
print(df_pred)

# Compute standard performance metrics of the linear regression:

# Mean Absolute Error
mae = metrics.mean_absolute_error(y_test_m, y_pred)
# Mean Squared Error
mse = metrics.mean_squared_error(y_test_m, y_pred)
# Root Mean Square Error
rmse =  math.sqrt(metrics.mean_squared_error(y_test_m, y_pred))
# Normalised Root Mean Square Error
y_max = y_mobile_updrs.max()
y_min = y_mobile_updrs.min()
rmse_norm = rmse / (y_max - y_min)

# R-Squared
r_2 = metrics.r2_score(y_test_m, y_pred)
Adj_r2 = 1 - (1-r_2) * (len(y_mobile_updrs)-1)/(len(y_mobile_updrs)-x.shape[1]-1)
print("MLP performance:")
print("MAE: ", mae)
print("MSE: ", mse)
print("RMSE: ", rmse)
print("RMSE (Normalised): ", rmse_norm)
print("R^2: ", r_2)
print("Adjusted R^2", Adj_r2)

# Performance metrics and their corresponding values
metric = ['MAE', 'MSE', 'RMSE', 'RMSE (Normalized)', 'R^2', 'Adjusted R^2']
values = [mae, mse, rmse, rmse_norm, r_2, Adj_r2]

# Create a bar graph with values on top
plt.figure(figsize=(8, 6))
bars = plt.bar(metric, values, color='skyblue')

# Display the values on top of the bars
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2 - 0.15, bar.get_height() + 0.2, round(value, 2), ha='center')

plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('MLP Performance Metrics')
plt.ylim(0, max(values) + 10)  # Setting y-axis limit for better visualization
plt.show()


# In[49]:


# Split dataset into 60% training and 40% test sets 
# Note: other % split can be used.
X_train, X_test, y_train_t, y_test_t = train_test_split(std_x_df, y_total_updrs, test_size=0.4, random_state=0)

# Build a linear regression model
model = LinearRegression()
# Train (fit) the linear regression model using the training set
model.fit(X_train, y_train_t)

# Print the intercept and coefficient learned by the linear regression model
print("Intercept: ", model.intercept_)
print("Coefficient: ", model.coef_)

# Use linear regression to predict the values of (y) in the test set
# based on the values of x in the test set
y_pred = model.predict(X_test)

# Optional: Show the predicted values of (y) next to the actual values of (y)
df_pred = pd.DataFrame({"Actual": y_test_t, "Predicted": y_pred})
print(df_pred)

# Compute standard performance metrics of the linear regression:

# Mean Absolute Error
mae = metrics.mean_absolute_error(y_test_t, y_pred)
# Mean Squared Error
mse = metrics.mean_squared_error(y_test_t, y_pred)
# Root Mean Square Error
rmse =  math.sqrt(metrics.mean_squared_error(y_test_t, y_pred))
# Normalised Root Mean Square Error
y_max = y_total_updrs.max()
y_min = y_total_updrs.min()
rmse_norm = rmse / (y_max - y_min)

# R-Squared
r_2 = metrics.r2_score(y_test_t, y_pred)
Adj_r2 = 1 - (1-r_2) * (len(y_total_updrs)-1)/(len(y_total_updrs)-x.shape[1]-1)
print("MLP performance:")
print("MAE: ", mae)
print("MSE: ", mse)
print("RMSE: ", rmse)
print("RMSE (Normalised): ", rmse_norm)
print("R^2: ", r_2)
print("Adjusted R^2", Adj_r2)

# Performance metrics and their corresponding values
metric = ['MAE', 'MSE', 'RMSE', 'RMSE (Normalized)', 'R^2', 'Adjusted R^2']
values = [mae, mse, rmse, rmse_norm, r_2, Adj_r2]

# Create a bar graph with values on top
plt.figure(figsize=(8, 6))
bars = plt.bar(metric, values, color='skyblue')

# Display the values on top of the bars
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2 - 0.15, bar.get_height() + 0.2, round(value, 2), ha='center')

plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('MLP Performance Metrics')
plt.ylim(0, max(values) + 10)  # Setting y-axis limit for better visualization
plt.show()


# # Work on previous dataset

# In[50]:


a = np.loadtxt('po1_data.txt',  delimiter = ',')


print(a)

df_old = pd.DataFrame(a)


df_old.columns = ['subject identifier', ' jitter in %', 'absolute jitter in ms', 'jitter r.a.p.', 'jitter p.p.q.5','jitter d.d.p.' ,
              'shimmer in %','absolute shimmer dB', 'shimmer a.p.q.3', 'shimmer a.p.q.5','shimmer a.p.q.11','shimmer d.d.a', 
              'autocorrelation between NHR and HNR', 'NHR','HNR','median pitch','mean pitch','sd of pitch','min pitch',
              'max pitch','number of pulses','number of periods','mean period','sd of period','fraction of unvoiced frames',
              'num of voice breaks','degree of voice breaks','UPDRS','PD label']


print(df_old)

column_names = list(df_old.columns)


df1 = df_old[df_old["PD label"] == 1]
df2 = df_old[df_old["PD label"] == 0]

salient_features = []
for i in range(1,28):
    print('Analysis of the measurement variable', column_names[i])
    print()
    sample1 = df1.iloc[:, i].to_numpy()
    sample2 = df2.iloc[:, i].to_numpy()
    
    

    # the basic statistics of sample 1:
    x_bar1 = st.tmean(sample1)
    s1 = st.tstd(sample1)
    n1 = len(sample1)
    print("\n Statistics of sample 1: %.3f (mean), %.3f (std. dev.), and %d (n)." % (x_bar1, s1, n1))
  
    # the basic statistics of sample 2:
    x_bar2 = st.tmean(sample2)
    s2 = st.tstd(sample2)
    n2 = len(sample2)
    print("\n Statistics of sample 2: %.3f (mean), %.3f (std. dev.), and %d (n)." % (x_bar2, s2, n2))
    

    # perform two-sample t-test
    # null hypothesis: mean of sample 1 = mean of sample 2
    # alternative hypothesis: mean of sample 1 is not equal to mean of sample 2
    # note the argument equal_var=False, which assumes that two populations do not have equal variance
    t_stats, p_val = st.ttest_ind_from_stats(x_bar1, s1, n1, x_bar2, s2, n2, equal_var=False, alternative='two-sided')
    print("\n Computing t* ...")
    print("\t t-statistic (t*): %.2f" % t_stats)

    print("\n Computing p-value ...")
    print("\t p-value: %.4f" % p_val)

    print("\n Conclusion:")
    if p_val < 0.05:
        print("\t We reject the null hypothesis for", column_names[i])
        salient_features.append(column_names[i])
    else:
        print("\t We accept the null hypothesis for", column_names[i])
        print()

        
print(salient_features)



# We only use df1 for our linear regression model as they are ppd

# In[51]:


x = df1[[' jitter in %', 'absolute jitter in ms', 'jitter r.a.p.', 'jitter p.p.q.5', 
         'jitter d.d.p.', 'shimmer a.p.q.11', 'autocorrelation between NHR and HNR', 
         'NHR', 'median pitch', 'mean pitch', 'sd of pitch', 'max pitch', 'mean period',
         'fraction of unvoiced frames', 'num of voice breaks', 
         'degree of voice breaks']] 
y = df1['UPDRS']


# Split dataset into 60% training and 40% test sets 
# Note: other % split can be used.
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)

# Build a linear regression model
model = LinearRegression()
# Train (fit) the linear regression model using the training set
model.fit(X_train, y_train)

# Print the intercept and coefficient learned by the linear regression model
print("Intercept: ", model.intercept_)
print("Coefficient: ", model.coef_)

# Use linear regression to predict the values of (y) in the test set
# based on the values of x in the test set
y_pred = model.predict(X_test)

# Optional: Show the predicted values of (y) next to the actual values of (y)
df_pred = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
print(df_pred)

# Compute standard performance metrics of the linear regression:

# Mean Absolute Error
mae = metrics.mean_absolute_error(y_test, y_pred)
# Mean Squared Error
mse = metrics.mean_squared_error(y_test, y_pred)
# Root Mean Square Error
rmse =  math.sqrt(metrics.mean_squared_error(y_test, y_pred))
# Normalised Root Mean Square Error
y_max = y.max()
y_min = y.min()
rmse_norm = rmse / (y_max - y_min)

# R-Squared
r_2 = metrics.r2_score(y_test, y_pred)
Adj_r2 = 1 - (1-r_2) * (len(y_mobile_updrs)-1)/(len(y_mobile_updrs)-x.shape[1]-1)
print("MLP performance:")
print("MAE: ", mae)
print("MSE: ", mse)
print("RMSE: ", rmse)
print("RMSE (Normalised): ", rmse_norm)
print("R^2: ", r_2)
print("Adjusted R^2", Adj_r2)

# Performance metrics and their corresponding values
metric = ['MAE', 'MSE', 'RMSE', 'RMSE (Normalized)', 'R^2', 'Adjusted R^2']
values = [mae, mse, rmse, rmse_norm, r_2, Adj_r2]

# Create a bar graph with values on top
plt.figure(figsize=(8, 6))
bars = plt.bar(metric, values, color='skyblue')

# Display the values on top of the bars
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2 - 0.15, bar.get_height() + 0.2, round(value, 2), ha='center')

plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('MLP Performance Metrics')
plt.ylim(0, max(values) + 10)  # Setting y-axis limit for better visualization
plt.show()


# # Now we do one sample t test on our new dataset to see the salient variables in the same way as before.

# In[52]:


df = pd.read_csv("po2_data.csv")

x = df[['age', 'sex', 'jitter(%)', 
              'jitter(abs)', 'test_time', 'jitter(rap)', 'jitter(ppq5)',
              'jitter(ddp)', 'shimmer(%)', 'shimmer(abs)', 
              'shimmer(apq3)', 'shimmer(apq5)', 'shimmer(apq11)', 
              'shimmer(dda)', 'nhr', 'hnr', 'rpde', 'dfa', 'ppe']]

y_mobile_updrs = df['motor_updrs']
y_total_updrs = df['total_updrs']

column_names = list(x.columns)

salient_features = []

for i in range(0,19):
    print('Analysis of the measurement variable', column_names[i])
    print()
    # sample values
    sample = x.iloc[:, i].to_numpy()

    # compute mean and standard deviation of the sample
    print("Computing the basic statistics ...")
    x_bar = st.tmean(sample)
    s = st.tstd(sample)
    print("\t Sample mean: %.2f" % x_bar)
    print("\t Sample std. dev.: %.2f" % s)

    # perform one-sample t-test
    t_stats, p_val = st.ttest_1samp(sample, x_bar, alternative='greater')
    print("\n Computing t* ...")
    print("\t t-statistic (t*): %.2f" % t_stats)

    print("\n Computing p-value ...")
    print("\t p-value: %.4f" % p_val)

    print("\n Conclusion:")
    if p_val < 0.05:
        print("\t We reject the null hypothesis for", column_names[i])
        salient_features.append(column_names[i])
    else:
        print("\t We accept the null hypothesis for", column_names[i])
        print()

        
print(salient_features)


# In[53]:


print(salient_features)


# In[ ]:




