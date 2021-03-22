import EDA
import polynomial_regression
import regression
from multiapp import MultiApp
import streamlit as st

app = MultiApp()
app.add_app("Exploratory Data Analysis", EDA.app)
app.add_app("Regression Analysis", regression.app)
app.add_app("Polynomial Regression Analysis", polynomial_regression.app)
st.title('Price Suggestion and Elasticity of Demand Analysis in E-commerce')
app.run()