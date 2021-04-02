import polynomial_regression
import regression
import overall_EDA
import product_EDA
import PED

from multiapp import MultiApp
import streamlit as st


app = MultiApp()

app.add_app("Overall Exploratory Data Analysis", overall_EDA.app)
app.add_app("Product Exploratory Data Analysis", product_EDA.app)
app.add_app("Regression Analysis", regression.app)
app.add_app("Polynomial Regression Analysis", polynomial_regression.app)
app.add_app("PED Comparison", PED.app)

st.title('Price Suggestion and Elasticity of Demand Analysis in E-commerce')
app.run()