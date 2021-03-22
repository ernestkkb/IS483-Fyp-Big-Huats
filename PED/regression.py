import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.pyplot import figure
import streamlit as st

from statsmodels.compat import lzip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols

#Function Definition
@st.cache(ttl = 3600)
def read_file(filename):
    data = pd.read_csv(filename, encoding = 'ISO-8859-1')
    return data

def display_regression_plot(data):
    g = sns.pairplot(data, kind = 'reg', size = 10, x_vars = ['UnitPrice'], y_vars = ['Quantity'])
    st.pyplot()

def partial_regression_plot(model):
    fig = plt.figure(figsize=(20,14))
    fig = sm.graphics.plot_partregress_grid(model)
    st.pyplot()
    st.subheader('Partial Regression Plot')
    st.text('''
    As seen in the regression plot on the right, price & quantity have exhibits a negative 
    relationship. When the price increases, the quantity of products sold will decrease.
    ''')

def ccpr_plot(model):
    fig = plt.figure(figsize=(12,8))
    fig = sm.graphics.plot_ccpr_grid(model, fig = fig)
    st.pyplot()
    st.subheader('Component-Component Plus Residual Plot')
    st.text('''
    To properly evaluate the effects of price on quantity, we have to take into account the 
    influence of other independent variables. As shown, the relationship between the variation
    in quantity explained by price is linear. There are not evident factors that are exerting 
    considerable influence on the linear relationship.
    ''')

def reg_plot(data):
    model = ols("np.log(Quantity) ~ np.log(UnitPrice)", data = data).fit()
    fig = plt.figure(figsize=(12,8))
    fig = sm.graphics.plot_regress_exog(model, 'np.log(UnitPrice)', fig=fig)
    st.pyplot()
    st.text('''
    In graph Y and fitted vs X & Residuals vs Price, we can see that the variation between 
    the predicted & the actual values are moderately high. The model is currently able to 
    only able to explain a proportion of the variance.''')


def ols_summary_log(data, newDict, description):
    model = ols("np.log(Quantity) ~ np.log(UnitPrice)", data = data).fit()
    st.text(model.summary())
    coefficient = model.params[1]
    r2 = model.rsquared
    st.subheader('Summary Table')
    st.text('''
    The small p values indicate that we can reject the null hypothesis that price has no effect
    on quantity. Therefore, price does have an effect on quantity for this particular item. 
    The coefficient of price is {:.3}. This refers to the inverse relationship between price and 
    quantity. When price increases, quantity demanded decreases & vice versa.However, our current
    model can only explain a proportion of the variance. This can be seen from its R score of 
    {:.3}. When price increases, quantity demanded decreases & vice versa.
    '''.format(coefficient, r2))

    newDict['Item'].append(description)
    newDict['R Squared Value'].append(model.rsquared)

    partial_regression_plot(model)
    ccpr_plot(model)


def rls_summary_log(data):
    endog = np.log(data['Quantity'])
    exog = sm.add_constant(np.log(data['UnitPrice']))
    mod = sm.RecursiveLS(endog, exog)
    res = mod.fit()
    st.text(res.summary())
    fig = res.plot_recursive_coefficient(range(mod.k_exog), alpha=None, figsize=(10,6));
    st.text('''The R-squared value, coefficient of price together and the p value remained consistent 
    with the Ordinary Least Squares (OLS) calculation.''')
    st.pyplot()

def PED(data):
    
    newDict = {
        'Item': [],
        'Price Elasticity of Demand':[]
    }
    
    chosen = ['23166', '85099B', '85123A', '23084', '22197']
    
    for stockID in chosen:
        data_modified = data[data['StockCode'] == stockID]

        model = ols("np.log(Quantity) ~ np.log(UnitPrice)", data = data_modified).fit()
        intercept, slope = model.params
        mean_price = np.mean(data_modified['UnitPrice'])
        mean_quantity = np.mean(data_modified['Quantity'])

        price_elasticity = abs((slope) * (mean_price/mean_quantity))

        newDict['Item'].append(data_modified.iloc[0]['Description'])
        newDict['Price Elasticity of Demand'].append(price_elasticity)
        
    return pd.DataFrame(newDict)

def ols_summary_log1(data):
    model = ols("np.log(Quantity) ~ UnitPrice", data = data).fit()
    st.text(model.summary())

def PED1(data): 
    newDict = {
        'Item': ['JBL Clip2 Portable Speaker'],
        'Price Elasticity of Demand':[]
    }
    
    model = ols("np.log(Quantity) ~ UnitPrice", data = data).fit()
    intercept, slope = model.params
    mean_price = np.mean(data['UnitPrice'])
    mean_quantity = np.mean(data['Quantity'])
    price_elasticity = ((slope) * (mean_price/mean_quantity))
    
    newDict['Price Elasticity of Demand'] = price_elasticity
    
    return pd.DataFrame(newDict)
@st.cache(ttl = 3600, suppress_st_warning=True)
def app():
    st.title('Regression Analysis')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    data = read_file('data.csv')
    data = data[(data['Quantity']> 0) & (data['UnitPrice'] > 0)]
    data['InvoiceDate']= pd.to_datetime(data['InvoiceDate']) #converting column invoicedate to datetime format
    data = data.set_index('InvoiceDate') #setting date as an index for the dataframe
    data['Year'] = data.index.year
    data['Month'] = data.index.month
    data['Weekday Name'] = data.index.day_name()
    data['Hour'] = data.index.hour

    #Remove Outlier
    from scipy import stats
    data = data[(np.abs(stats.zscore(data['Quantity'])) < 3)]

    newDict = {
            'Item': [],
            'R Squared Value':[]
        }

    st.header('Choice of Regression Models')

    st.subheader('Ordinary Least Squares Regression')
    st.text('''
    Ordinary Least Squares (OLS) regression is a statistical method of analysis that estimates 
    the relationship between one or more independent variables and a dependent variable. It 
    estimates the aforementioned relationship by minimizing the sum of  squares in the 
    difference between the observed and predicted values of the dependent variable configured 
    as a straight line.
    ''')
    st.subheader('Recursive Least Squares Regression')
    st.text('''
    Recursive least squares (RLS) is an expanding window version of ordinary least squares. 
    In addition to recursively calculating the regression coefficients, it aims to minimize
    a weighted linear least squares cost function rather than the usual mean square error.
    ''')
    st.subheader('Polynomial Regression')
    st.text('''
    The motivation behind using 2 models is to check for any possible discrepancy in our 
    regression analysis results and ascertain if our results are consistent & validated.
    ''')

    st.header('Regression Analysis for Household Items')
    chosen = {
        '23166':'Medium Ceramic Storage Jar',
        '85099B':'Jumbo Bag Red Retrospot',
        '85123A':'White Hanging Heart T-Light Holder',
        '23084':'Rabbit Night Light',
        '22197':'Small Popcorn Holder'
    }

    for id in chosen:
        product_title = chosen[id]
        st.header("Household Product: "+product_title)
        st.subheader('Regression Plot')
        display_regression_plot(data[data['StockCode'] == id])
        st.subheader('Ordinary Least Squares (OLS) Estimation')
        ols_summary_log(data[data['StockCode'] == id], newDict, product_title)
        st.subheader('Regression Plot')
        reg_plot(data[data['StockCode'] == id])
        st.subheader('Rercursive Least Squares (RLS) Estimation')
        rls_summary_log(data[data['StockCode'] == id])
        st.subheader('Conclusion for '+product_title)
        st.text('''Both Ordinary Least Squares (OLS) & Recursive Least Squares (RLS) are consistent 
    with the fact that price exhibits a OPPOSITE influence on quantity. Since the price 
    elasticity of demand is < 1, we can conclude that the item is inelastic where changes
    in price have a less than proportional effect on the quantity of the good demanded.
    ''')
    st.subheader('Conclusion for all household items')
    st.text('''
    Household items sold on ecommerce platforms tend to be inelastic where changes in price
    have a less than proportional effect on the quantity of the good demanded. This can be
    seen from the aforementioned 5 household items where we concluded that price had a
    minimal change in quantity of products sold. This means that consumers are not sensitive
    towards the changes in price (they buy about the same quantity even when prices change). 
    Furthermore, the relationship between price and quantity of household products is negative
    ''')

    st.subheader('Price Elasticity of Demand')

    r2_df = pd.DataFrame(newDict)
    st.dataframe(r2_df)
    plt.figure(figsize=(25, 10))
    plt.bar(r2_df['Item'], r2_df['R Squared Value'], color=(0.2, 0.4, 0.6, 0.8),  width = 0.4) 
    st.pyplot()

    ped_df = PED(data)
    st.dataframe(ped_df)
    plt.figure(figsize=(25, 10))
    fig = plt.bar(ped_df['Item'], ped_df['Price Elasticity of Demand'], color=(0.2, 0.4, 0.6, 0.8),  width = 0.4)
    st.pyplot()

    st.text('''
    While the R squared value is low, the low p values still indicate a real relationship
    between price & quantity of products sold. The only caveat here is models with lower 
    R square have a larger prediction interval. This would result in predictions that 
    may not be as precise for a given regression equation. However, since we are not using
    the model to predict price or quantity for a given product. The R squared value is not
    taken with as much importance as we can already ascertain that the products' elasticity
    through the changes in quantity & price.''')

    st.subheader('''To further improve the r squared value of the household products, 
    we will attempt to improve its complexity via polynomial regression in a separate section.''')

    st.header('Regression Analysis for Electronic Products')
    df = read_file('category_price.csv')
    df = df[df['condition'] == 'New']
    #Drop columns that are not going to be used for the project
    df = df[df.columns.drop(['condition'])]
    #Using only USD prices
    df_us = df[df['currency'] == 'USD']

    #Filter only the impressions with price variation for further analysis
    # at least 5 different prices
    df_detect = df_us.groupby('name')['disc_price'].nunique()
    df_detect = df_detect.to_frame().reset_index()
    df_detect.columns = ['name','disc_price']
    df_detect

    valid_items = df_detect[df_detect['disc_price'] > 5]
    valid_items = valid_items['name'].tolist()

    #Print Deleted Impressions with price variability
    df_clean = df_us[df_us['name'].isin(valid_items)]
    #Detect top selling items
    top = df_clean.groupby('name').size().reset_index(name='counts').nlargest(15,['counts'])
    top_products = top['name'].tolist()
    df_clean = df_clean[df_clean['name'].isin(top_products)]
    frames = []
    for i in top_products:
        test = df_clean[df_clean.name.str.contains(i)]
        test = test[(np.abs(stats.zscore(test['disc_price'])) < 3)]
        frames.append(test)

    final_df = pd.concat(frames)

    #Drop columns that are not going to be used for the project
    df_clean = df_clean[['name', 'price', 'disc_price', 'merchant', 'brand', 
            'Category_name', 'Day_n', 'year', 'month', 'month_n','day', 'Week_Number']]

    testing = final_df.loc[final_df['name'] == 'JBL Clip2 Portable Speaker']
    tester = testing.groupby(['disc_price']).size()
    test_df = tester.to_frame().reset_index()
    test_df.columns = ['UnitPrice','Quantity']
    st.subheader('Regression Plot')
    display_regression_plot(test_df)
    st.subheader('Ordinary Least Squares (OLS) Estimation')
    ols_summary_log1(test_df)
    st.subheader('Rercursive Least Squares (RLS) Estimation')
    rls_summary_log(test_df)
    st.dataframe(PED1(test_df))