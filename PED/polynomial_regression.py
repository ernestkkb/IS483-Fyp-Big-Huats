import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.pyplot import figure
import streamlit as st
import statsmodels.api as sm
import pathlib

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff

from bokeh.plotting import figure, output_notebook, show
from bokeh.models.tools import HoverTool
from bokeh.io import output_notebook
from bokeh.models import ColumnDataSource
from bokeh.palettes import cividis, inferno,Spectral3,Spectral4,Spectral5,Spectral6, Spectral7,Category20c
from bokeh.plotting import figure
from bokeh.transform import cumsum
################################################################################################################################ Function Definitions ####################################################################################
@st.cache(allow_output_mutation=True)
def read_file(filename):

    if(filename == "data.csv"):
        data = pd.read_csv(filename, encoding = 'ISO-8859-1')

    else:
        data = pd.read_csv(filename)

    return data

def create_polynomial_regression_model(degree, X_train, X_test, y_train, y_test):
    st.subheader('Linear to Quadratic Transformation')
    with st.echo():
        poly_features = PolynomialFeatures(degree=degree)

        # transforms the existing features to higher degree features.
        X_train_poly = poly_features.fit_transform(X_train)

        # fit the transformed features to Linear Regression
        poly_model = LinearRegression()
        poly_model.fit(X_train_poly, y_train)

        # predicting on training data-set
        y_train_predicted = poly_model.predict(X_train_poly)

        # predicting on test data-set
        y_test_predict = poly_model.predict(poly_features.fit_transform(X_test))

        # evaluating the model on training dataset
        rmse_train = np.sqrt(mean_squared_error(y_train, y_train_predicted))
        r2_train = r2_score(y_train, y_train_predicted)

        # evaluating the model on test dataset
        rmse_test = np.sqrt(mean_squared_error(y_test, y_test_predict))
        r2_test = r2_score(y_test, y_test_predict)
        
    slope = poly_model.coef_
    
    st.text("The model performance for the training set")
    st.text("-------------------------------------------")
    st.text("RMSE of training set is {}".format(rmse_train))
    st.text("R2 score of training set is {}".format(r2_train))

    st.text("\n")

    st.text("The model performance for the test set")
    st.text("-------------------------------------------")
    st.text("RMSE of test set is {}".format(rmse_test))
    st.text("R2 score of test set is {}".format(r2_test))
    
    st.text("\n")
    
    st.text("The slope is {}".format(slope[0][1]))

def statistics(degree, X_train, X_test, y_train, y_test):

    poly_features = PolynomialFeatures(degree=degree)

    # transforms the existing features to higher degree features.
    X_train_poly = poly_features.fit_transform(X_train)

    # fit the transformed features to Linear Regression
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)

    # predicting on training data-set
    y_train_predicted = poly_model.predict(X_train_poly)

    # predicting on test data-set
    y_test_predict = poly_model.predict(poly_features.fit_transform(X_test))

    # evaluating the model on training dataset
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_predicted))
    r2_train = r2_score(y_train, y_train_predicted)

    # evaluating the model on test dataset
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_predict))
    r2_test = r2_score(y_test, y_test_predict)
    
    slope = poly_model.coef_
    
    return [slope[0][1], rmse_train, r2_train, rmse_test, r2_test]

def display_regression_plot(data):
    fig = px.scatter(data, x="UnitPrice", y="Quantity", trendline="lowess")
    st.plotly_chart(fig, use_container_width=True)

def data_transformation_standardisation(data, stockID):
    data_selected = data[data['StockCode'] == stockID]

    #Transform Skewed Columns
    data_selected['Quantity'] = np.log(data_selected['Quantity'])
    data_selected['UnitPrice'] = np.log(data_selected['UnitPrice'])
    data_selected = data_selected[['Quantity','UnitPrice']]


    #Standardisation of Scales
    scaler = StandardScaler()
    scaler.fit(data_selected)
    scaler.fit_transform(data_selected)
    data_scaled= scaler.transform(data_selected)
    data_prepared=pd.DataFrame(data_scaled, columns=data_selected.columns)

    return data_prepared

def calculate_PED(data, y_train, X_train, X_test, y_test, description, stockID):

    mod = sm.OLS(y_train, X_train)
    m1 = mod.fit()
    p_values = m1.summary2().tables[1]['P>|t|']
    st.text("P Value: "+str(round(p_values.iloc[0], 4)))

    stats = statistics(2, X_train, X_test, y_train, y_test)
    slope = stats[0]

    mean_price = np.mean(data[data['StockCode'] == stockID]['UnitPrice'])
    mean_quantity = np.mean(data[data['StockCode'] == stockID]['Quantity'])
    price_elasticity = abs((slope) * (mean_price/mean_quantity))
    st.text("Price Elasticity of Demand: "+str(price_elasticity))

def calculate_r2(products, name_map1, data, newDict2):
    test_size = [0.4, 0.2, 0.2, 0.2, 0.2]
    i = 0
    for product in products:
        id = name_map1[product]
        data_prepared = data_transformation_standardisation(data,id)

        X = data_prepared[['UnitPrice']]
        y = data_prepared[['Quantity']]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size[i], random_state = 101)
        stats = statistics(2, X_train, X_test, y_train, y_test)
        newDict2['Item'].append(product)
        newDict2['R Squared Value'].append(stats[-1])
        i+=1

    return newDict2

def calculate_PED_all(products, name_map1, data, newDict):
    test_size = [0.4, 0.2, 0.2, 0.2, 0.2]
    i = 0

    for product in products:
        id = name_map1[product]
        data_prepared = data_transformation_standardisation(data,id)

        X = data_prepared[['UnitPrice']]
        y = data_prepared[['Quantity']]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size[i], random_state = 101)

        mod = sm.OLS(y_train, X_train)

        stats = statistics(2, X_train, X_test, y_train, y_test)
        slope = stats[0]

        mean_price = np.mean(data[data['StockCode'] == id]['UnitPrice'])
        mean_quantity = np.mean(data[data['StockCode'] == id]['Quantity'])
        price_elasticity = abs((slope) * (mean_price/mean_quantity))

        newDict['Item'].append(product)
        newDict['Price Elasticity of Demand'].append(price_elasticity)
        i+=1
    return newDict
################################################################################################################################ Function Definitions ####################################################################################
def app():
    data = read_file('data.csv')
    top_products1 = ['MEDIUM CERAMIC TOP STORAGE JAR', 'JUMBO BAG RED RETROSPOT', 'WHITE HANGING HEART T-LIGHT HOLDER', 'RABBIT NIGHT LIGHT', 'SMALL POPCORN HOLDER']
    name_map = {
        '23166' : 'MEDIUM CERAMIC TOP STORAGE JAR',
        '85099B': 'JUMBO BAG RED RETROSPOT',
        '85123A': 'WHITE HANGING HEART T-LIGHT HOLDER',
        '23084': 'RABBIT NIGHT LIGHT',
        '22197': 'SMALL POPCORN HOLDER'
    }

    name_map1 = {
        'MEDIUM CERAMIC TOP STORAGE JAR':'23166',
        'JUMBO BAG RED RETROSPOT':'85099B',
        'WHITE HANGING HEART T-LIGHT HOLDER':'85123A',
        'RABBIT NIGHT LIGHT':'23084',
        'SMALL POPCORN HOLDER':'22197'
    }

################################################################################################################################ Data Preprocessing for Household Products ####################################################################################
    data = data[(data['Quantity']> 0) & (data['UnitPrice'] > 0)]
    data = data.reindex(data.index.repeat(data.Quantity)) #multiply the quantity of items to expand the number of rows
    data['InvoiceDate']= pd.to_datetime(data['InvoiceDate']) #converting column invoice date to datetime format
    data = data.set_index('InvoiceDate') #setting date as an index for the dataframe

    #Adding additional time-based columns
    data['Year'] = data.index.year
    data['Month'] = data.index.month
    data['Weekday Name'] = data.index.day_name()
    data['Hour'] = data.index.hour

    #Remove Outlier
    from scipy import stats
    data = data[(np.abs(stats.zscore(data['Quantity'])) < 3)]
################################################################################################################################ Displaying of graphs ####################################################################################
    st.title('Polynomial Regression Analysis')
    st.write('''
    As our straight line is unable to capture the salient pattern in the data, we increased the complexity of the model to overcome the under-fitting issue.
    We did this by generating a higher order function from one that is linear to a quadratic one.
    To convert the original features into their higher order terms we will use the PolynomialFeatures class provided by scikit-learn. After which, we trained the model using Linear Regression.
    ''')
     
    newDict = {
            'Item': [],
            'Price Elasticity of Demand':[]
    }

    newDict2 = {
            'Item': [],
            'R Squared Value':[]
    }

    product = st.selectbox('Pick a product', options = top_products1, key = 1)

    id = name_map1[product]
    data_prepared = data_transformation_standardisation(data,id)

    X = data_prepared[['UnitPrice']]
    y = data_prepared[['Quantity']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 101)
    X_train.shape, y_train.shape, X_test.shape, y_test.shape
    create_polynomial_regression_model(2, X_train, X_test, y_train, y_test)
    display_regression_plot(data_prepared)
    calculate_PED(data, y_train, X_train, X_test, y_test, product, id)

    r2_df = pd.DataFrame(calculate_r2(top_products1, name_map1, data, newDict2))
    st.title("Improved R Squared Value Bar Plot")

    item = r2_df['Item']
    r2 = r2_df['R Squared Value']

    source = ColumnDataSource(data=dict(item=item, r2=r2, color=Spectral5))

    max_y = max(r2)

    p = figure(x_range=item, y_range=(0,max_y), plot_height=500, title="Improved R^2 Value of each Product",
        toolbar_location=None, tools="")
    
    p.vbar(x='item', top='r2', width=0.9, color='color', source=source)
    
    p.xgrid.grid_line_color = None
    p.legend.orientation = "horizontal"
    p.legend.location = "top_center"
    p.xaxis.visible = False

    p.add_tools(HoverTool(tooltips=[("Product: ", "@item"),("R2 value:", "@r2")]))

    st.bokeh_chart(p, use_container_width=True)
    st.dataframe(r2_df)

    
    ped_df = pd.DataFrame(calculate_PED_all(top_products1, name_map1, data, newDict))
    st.title("PED Bar Plot")

    item = ped_df['Item']
    ped = ped_df['Price Elasticity of Demand']

    source = ColumnDataSource(data=dict(item=item, ped=ped, color=Spectral5))

    max_y = max(ped)

    p = figure(x_range=item, y_range=(0,max_y), plot_height=500, title="Updated PED value of each Product",
        toolbar_location=None, tools="")
    
    p.vbar(x='item', top='ped', width=0.9, color='color', source=source)
    
    p.xgrid.grid_line_color = None
    p.legend.orientation = "horizontal"
    p.legend.location = "top_center"
    p.xaxis.visible = False

    p.add_tools(HoverTool(tooltips=[("Product: ", "@item"),("PED value:", "@ped")]))

    st.bokeh_chart(p, use_container_width=True)
    st.dataframe(ped_df)

    st.text('''
    Since the price elasticity of demand is still < 1 for all household products, 
    we can conclude that the product is inelastic where changes in price have a 
    less than proportional effect on the quantity of the good demanded.
    ''')