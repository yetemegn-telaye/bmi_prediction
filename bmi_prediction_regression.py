#!/usr/bin/env python
# coding: utf-8

# # Importing all the dependencies

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import joblib
import streamlit as st


# # Import the dataset

# In[44]:


df = pd.read_csv('gym_members_exercise_tracking.csv')
data = df.head()


# In[43]:


data = df.tail()


# # Exploratory Data Analysis (EDA)

# ### Understand the distribution, relationship and trends

# In[48]:


data = df.describe().T


# In[47]:


data = df.describe(include="object")


# In[38]:


data = df.info()


# In[85]:


for col in df.columns:
    print(col, len(df[col].unique()) , df[col].unique())


# In[36]:


data = df.dtypes


# ### Visualize the dataset using charts and plots

# - unvarient analysis for numerical analysis

# In[91]:


for i in df.select_dtypes(include="number").columns:
    sns.histplot(data=df,x=i)
    plt.show()


# - unvarient analysis for categorical analysis

# In[89]:


for i in df.select_dtypes(include="object").columns:
    sns.countplot(data=df,x=i)
    plt.show()


# In[94]:


for i in df.select_dtypes(include="object").columns:
    sns.boxplot(data=df,x=i)
    plt.show()


# In[95]:


for i in df.select_dtypes(include="number").columns:
    sns.boxplot(data=df,x=i)
    plt.show()


# - bivariant for numerical vs numerical (correlation matrix heatmap or scatter plot)

# In[96]:


df.select_dtypes(include="number").columns


# In[97]:


for i in ['Age', 'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM', 'Resting_BPM',
       'Session_Duration (hours)', 'Calories_Burned', 'Fat_Percentage',
       'Water_Intake (liters)', 'Workout_Frequency (days/week)',
       'Experience_Level']:
    sns.scatterplot(data=df,x=i,y='BMI')
    plt.show()


# In[99]:


df_corr = df.select_dtypes(include="number").corr()


# In[105]:


plt.figure(figsize=(15,15))
sns.heatmap(df_corr,annot=True)


# # Data Cleaning

# In[107]:


data_shape = df.shape


# In[108]:


data = df.info()


# - handle missing values

# In[110]:


data = df.isnull().sum()


# In[111]:


data = df.isnull().sum()/df.shape[0]*100


# - handle duplicates

# In[113]:


data = df.duplicated().sum()


# - Find garbage values

# In[114]:


for i in df.select_dtypes(include="object").columns:
    print(df[i].value_counts())
    print("***"*10)


# - address outliers and dependecies

# In[115]:


for i in df.select_dtypes(include="number").columns:
    sns.boxplot(data=df,x=i)
    plt.show()


# # Feature Engineering

# - Encode categorical variable

# In[5]:


data = df.select_dtypes(include="object").columns


# In[6]:


encoded_df = pd.get_dummies(df, columns=['Gender'], drop_first=False)


# In[7]:


new_encoded_df = pd.get_dummies(encoded_df, columns=['Workout_Type'], drop_first=False)
data=new_encoded_df.head()


# - Scale or normalize numerical features

# In[8]:


data = new_encoded_df


# In[23]:


data = df.columns


# In[24]:


df=new_encoded_df
numericals = ['Age', 'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM', 'Resting_BPM',
       'Session_Duration (hours)', 'Calories_Burned', 'Fat_Percentage',
       'Water_Intake (liters)', 'Workout_Frequency (days/week)',
       'Experience_Level', 'Gender_Female', 'Gender_Male',
       'Workout_Type_Cardio', 'Workout_Type_HIIT', 'Workout_Type_Strength',
       'Workout_Type_Yoga']
scaler = MinMaxScaler()
df[numericals] = scaler.fit_transform(df[numericals])
joblib.dump(scaler, 'scaler.pkl')
data = df


# # Data Splitting

# In[25]:


x= df.drop('BMI',axis=1)
y=df['BMI']
X_train, X_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
print(f"Train size: {X_train.shape[0]}, Validation size: {X_val.shape[0]}, Test size: {X_test.shape[0]}")


# # Model Development and Training

# In[26]:


#select a model and initialize
model = LinearRegression()


# In[27]:


#train the model on training set
model.fit(X_train,y_train)


# # Model Evaluation and hyperparameter tunning

# In[28]:


#validate the model on validation set
y_val_pred = model.predict(X_val)
#evaluate performance on validation set
val_mae = mean_absolute_error(y_val,y_val_pred)
val_mse = mean_squared_error(y_val,y_val_pred)
var_r2_score = r2_score(y_val,y_val_pred)
print(f"Validation on mean absoulte error: {val_mae:.2f}")
print(f"Validation on mean squared error: {val_mse:.2f}")
print(f"Validation on r2 score: {var_r2_score:.2f}")


# - check for overfitting and underfitting

# In[29]:


# Predict on the training set
y_train_pred = model.predict(X_train)

# Evaluate on the training set
train_mae = mean_absolute_error(y_train, y_train_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

print(f"Validation Performance:")
print(f"Validation mean absoulte error: {val_mae:.2f}")
print(f"Validation mean squared error: {val_mse:.2f}")
print(f"Validation r2 score: {var_r2_score:.2f}")

print(f"Training Performance:")
print(f"Train mean absoulte error: {train_mae:.2f}")
print(f"Train mean squared error: {train_mse:.2f}")
print(f"Train r2 score: {train_r2:.2f}")


# # Model testing and deployment

# In[30]:


y_test_pred = model.predict(X_test)
# evaluate performance on test set
test_mae = mean_absolute_error(y_test,y_test_pred)
test_mse = mean_squared_error(y_test,y_test_pred)
test_r2 = r2_score(y_test,y_test_pred)

print(f"Testing Performance:")
print(f"Test mean absoulte error: {test_mae:.2f}")
print(f"Test mean squared error: {test_mse:.2f}")
print(f"Test r2 score: {test_r2:.2f}")


# In[21]:


joblib.dump(model, 'bmi_predicting_model.pkl')
print("Model saved as 'bmi_predicting_model.pkl'")


# In[22]:


df.columns


# In[35]:


warnings.filterwarnings('ignore',category=FutureWarning)
model = joblib.load('bmi_predicting_model.pkl')
st.title('BMI Prediction App')
st.header('Enter Features for BMI Prediction')
age = st.number_input('Age',value=0.0)
weight = st.number_input('Weight(Kg)',value=0.0)
height = st.number_input('Height(cm)',value=0.0)
Max_BPM = st.number_input('Maximum BPM',value=0.0)
Avg_BPM = st.number_input('Avg BPM',value=0.0)
Resting_BPM = st.number_input('Resting BPM',value=0.0)
Session_Duration = st.number_input('Workout Session duration',value=0.0)
Calories_Burned = st.number_input('Calories Burned',value=0.0)
Fat_Percentage = st.number_input('Fat Percentage',value=0.0)
Water_Intake = st.number_input('Water intake(liters)',value=0.0)
Workout_Frequency = st.number_input('Workout Frequency(days/week)',value=0.0)
Experience_Level = st.number_input('Experience Level',value=0.0)

gender = st.selectbox('Gender',['Female','Male'])
gender_encoded = {
    'Gender_Female': 0 ,
    'Gender_Male': 0
}
gender_encoded[f'Gender_{gender}'] = 1

workout_type = st.selectbox('Workout Type',['Cardio','HIIT','Strength','Yoga'])
workout_encoded = {
    'Workout_Type_Cardio': 0,
    'Workout_Type_HIIT': 0,
    'Workout_Type_Strength': 0,
    'Workout_Type_Yoga': 0
}
workout_encoded[f'Workout_Type_{workout_type}'] = 1

if st.button('Predict BMI'):
    input_data = pd.DataFrame({
        'Age':[age],
        'Weight (kg)':[weight],
        'Height (m)':[height],
        'Max_BPM':[Max_BPM],
        'Avg_BPM': [Avg_BPM],
        'Resting_BPM': [Resting_BPM],
        'Session_Duration (hours)': [Session_Duration],
        'Calories_Burned': [Calories_Burned],
        'Fat_Percentage': [Fat_Percentage],
        'Water_Intake (liters)': [Water_Intake],
        'Workout_Frequency (days/week)':[ Workout_Frequency],
        'Experience_Level': [Experience_Level],
        **gender_encoded,
        **workout_encoded
    })
    input_data_scaled = scaler.fit_transform(input_data)
    prediction = model.predict(input_data_scaled)
    if isinstance(prediction, (list, np.ndarray)):
        prediction_value = prediction[0] 
    else:
        prediction_value = prediction 
    prediction = max(prediction_value, 0) 
    st.success(f'Predicted BMI: {prediction_value:.2f}')


# In[ ]:





# In[ ]:




