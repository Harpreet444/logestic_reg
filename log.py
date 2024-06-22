import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import joblib


st.set_page_config(page_title='Logestic reg',page_icon = '📈',layout='wide')
df = pd.read_csv("HR_comma_sep.csv")

left = df[df.left==1].drop(['Department','salary','left'],axis='columns')

retained = df[df.left==0].drop(['Department','salary','left'],axis='columns')

col1,col2 = st.columns([1,1])

col1.subheader('Employee lefted')
col1.table(left.mean())
col2.subheader('Employee retained')
col2.table(retained.mean())

# col,c = st.columns([1])
st.write('''Finding: Retained employees has higher satisfaction_level and lower average montly work hours, also have higher average promotion
         so we can safely drop (last_evaluation, number_project, time_spend_company) columns from data set.''')



co1,co2 = st.columns([1,1])
# Plot
salary_retention = df.groupby(['salary', 'left']).size().unstack(fill_value=0)

# Plot the grouped bar chart
fig, ax = plt.subplots(figsize=(10, 6))

# Define the bar width
bar_width = 0.35

# Define positions for the groups of bars
positions = np.arange(len(salary_retention))

# Plot bars for employees who stayed
ax.bar(positions - bar_width/2, salary_retention[0], bar_width, label='Stayed',color = 'g')

# Plot bars for employees who left
ax.bar(positions + bar_width/2, salary_retention[1], bar_width, label='Left', color = 'y')

# Set the title and labels
ax.set_title('Impact of Salaries on Employee Retention')
ax.set_xlabel('Salary Level')
ax.set_ylabel('Number of Employees')
ax.set_xticks(positions)
ax.set_xticklabels(salary_retention.index)

# Add legend
ax.legend()

# Display the plot in Streamlit
co1.pyplot(fig)



dep_retention = df.groupby(['Department', 'left']).size().unstack(fill_value=0)

# Plot the grouped bar chart
fig, ax = plt.subplots(figsize=(10, 6))

# Define the bar width
bar_width = 0.35

# Define positions for the groups of bars
positions = np.arange(len(dep_retention))

# Plot bars for employees who stayed
ax.bar(positions - bar_width/2, dep_retention[0], bar_width, label='Stayed',color = 'pink')

# Plot bars for employees who left
ax.bar(positions + bar_width/2, dep_retention[1], bar_width, label='Left', color = 'skyblue')

# Set the title and labels
ax.set_title('Impact of Department on Employee Retention')
ax.set_xlabel('Department')
ax.set_ylabel('Number of Employees')
ax.set_xticks(positions)
ax.set_xticklabels(dep_retention.index,)

# Add legend
ax.legend()

# Display the plot in Streamlit
co2.pyplot(fig)

df = df.drop(['last_evaluation','number_project','time_spend_company'],axis='columns')

st.write("Logestic regression model")
st.code("""
log_reg = reg.fit(x_train,y_train)
log_reg
        """)

# loading model and encodded objects
reg = joblib.load("log_reg.joblib")
ohe = joblib.load("ohe.joblib")
lab = joblib.load("label.joblib")

st.subheader('Model accuracy')
st.code(reg.score(pd.read_csv("x_test.csv"),pd.read_csv('y_test.csv')))
