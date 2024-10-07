# Employee Retention Analysis using Logistic Regression

This project uses logistic regression to analyze and predict employee retention based on various features. The analysis is visualized using a Streamlit app.

## Requirements

- streamlit
- pandas
- matplotlib
- numpy
- seaborn
- joblib

## Dataset

The dataset used is `HR_comma_sep.csv`, which contains information about employees, including their satisfaction level, last evaluation, number of projects, average monthly hours, time spent in the company, and more.

## Steps

1. **Data Preparation**: Load the dataset and preprocess it by dropping irrelevant columns and handling missing values.
2. **Feature Analysis**: Analyze the features that influence employee retention, such as satisfaction level, average monthly hours, and promotion.
3. **Model Training**: Train a logistic regression model to predict employee retention.
4. **Visualization**: Visualize the data and model predictions using Streamlit.

## Streamlit App

The Streamlit app provides an interactive interface for exploring the data and model predictions.

### Installation

To run the Streamlit app, you'll need to install the required packages:

```sh
pip install streamlit pandas matplotlib numpy seaborn joblib
