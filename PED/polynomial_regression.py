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


st.title('Polynomial Regression Analysis')
st.write('''
As our straight line is unable to capture the salient pattern in the data, we increased the complexity of the model to overcome the under-fitting issue.
We did this by generating a higher order function from one that is linear to a quadratic one.
To convert the original features into their higher order terms we will use the PolynomialFeatures class provided by scikit-learn. After which, we trained the model using Linear Regression.
''')
 

def create_polynomial_regression_model(degree, X_train, X_test, y_train, y_test):

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
    st.subheader("Regression Plot")
    fig = sns.pairplot(data, kind = 'reg', height = 10, x_vars = ['UnitPrice'], y_vars = ['Quantity'])
    st.pyplot(fig)


@st.cache
def read_file ():
    data = pd.read_csv('data.csv', encoding = 'ISO-8859-1')
    return data

def data_transformation_standardisation(data, stockID):
    data_selected = data[data['StockCode'] == stockID]
    st.dataframe(data_selected)

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
    st.subheader('After Transformation & Standardisation of Data')
    st.dataframe(data_prepared.head())

    return data_prepared

def calculate_PED(data, y_train, X_train, X_test, y_test, newDict, newDict2, description, stockID):

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
    newDict['Item'].append(description)
    newDict['Price Elasticity of Demand'].append(price_elasticity)
    newDict2['Item'].append(description)
    newDict2['R Squared Value'].append(stats[-1])


data = read_file()

# Remove noise data from quantity & UnitPrice 
data = data[(data['Quantity']> 0) & (data['UnitPrice'] > 0)]

#Remove Outliers
data = data[(np.abs(stats.zscore(data['Quantity'])) < 3)]

st.subheader("Data")
st.dataframe(data.head())
st.subheader("Main Code")
with st.echo():
    
    products = {
        'MEDIUM CERAMIC TOP STORAGE JAR':'23166', 
        'JUMBO BAG RED RETROSPOT':'85099B', 
        'WHITE HANGING HEART T-LIGHT HOLDER':'85123A', 
        'RABBIT NIGHT LIGHT':'23084', 
        'SMALL POPCORN HOLDER':'22197'}

    newDict = {
            'Item': [],
            'Price Elasticity of Demand':[]
        }

    newDict2 = {
            'Item': [],
            'R Squared Value':[]
        }
    for product_desc in products:
        description = product_desc
        stockID = products[product_desc]
        st.header(description)
        data_prepared = data_transformation_standardisation(data,stockID)
        X = data_prepared[['UnitPrice']]
        y = data_prepared[['Quantity']]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 101)
        X_train.shape, y_train.shape, X_test.shape, y_test.shape
        create_polynomial_regression_model(2, X_train, X_test, y_train, y_test)
        display_regression_plot(data_prepared)
        calculate_PED(data, y_train, X_train, X_test, y_test, newDict, newDict2, description, stockID)

ped_df = pd.DataFrame(newDict)
st.title("PED Bar Plot")
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ped = ped_df['Price Elasticity of Demand']
categories = ped_df['Item']
ax.bar(categories,ped)
plt.xticks(
    rotation=90, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large'  
)

st.pyplot(fig)

st.dataframe(ped_df)

r2_df = pd.DataFrame(newDict2)
st.title("R2 Bar Plot")
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
r2 = r2_df['R Squared Value']
categories = r2_df['Item']
ax.bar(categories,r2)
plt.xticks(
    rotation=90, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large'  
)
st.pyplot(fig)

st.dataframe(r2_df)

st.subheader("Helper Functions")
with st.echo():
    def create_polynomial_regression_model(degree, X_train, X_test, y_train, y_test):

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
        st.subheader("Regression Plot")
        fig = sns.pairplot(data, kind = 'reg', height = 10, x_vars = ['UnitPrice'], y_vars = ['Quantity'])
        st.pyplot(fig)


    @st.cache
    def read_file ():
        data = pd.read_csv('data.csv', encoding = 'ISO-8859-1')
        return data

    def data_transformation_standardisation(data, stockID):
        data_selected = data[data['StockCode'] == stockID]
        st.dataframe(data_selected)

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
        st.subheader('After Transformation & Standardisation of Data')
        st.dataframe(data_prepared.head())

        return data_prepared

    def calculate_PED(data, y_train, X_train, X_test, y_test, newDict, newDict2, description, stockID):

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
        newDict['Item'].append(description)
        newDict['Price Elasticity of Demand'].append(price_elasticity)
        newDict2['Item'].append(description)
        newDict2['R Squared Value'].append(stats[-1])