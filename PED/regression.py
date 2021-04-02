import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.pyplot import figure
import streamlit as st
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px

from statsmodels.compat import lzip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Image.MAX_IMAGE_PIXELS = None
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

def display_regression_plot(data):
    fig = px.scatter(data, x="UnitPrice", y="Quantity", trendline="ols")
    st.plotly_chart(fig, use_container_width=True)
    st.text('R Squared value in regression plot is not reflective as it is calculated before the transformation & standardisation of data')

def partial_regression_plot(model, flag):
    plt.rcParams["figure.figsize"] = (26,22)
    fig = sm.graphics.plot_partregress_grid(model)
    st.pyplot()
    st.subheader('Partial Regression Plot')

    if(flag):
        st.text('''As seen in the regression plot on the right, price & quantity have exhibits a negative 
relationship. When the price increases, the quantity of products sold will decrease.''')
    else:
        st.text('''As seen in the regression plot on the right, price & quantity have exhibits a positive 
relationship. When the price increases, the quantity of products sold will increase.''')

def ols_summary_log(data, description):
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

    partial_regression_plot(model, True)


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

def summary2(name, data):
        df = data[data['name'] == name]
        grouping = df.groupby(['disc_price']).size()
        grouping_df = grouping.to_frame().reset_index()
        grouping_df.columns = ['UnitPrice','Quantity']
        return pd.DataFrame(grouping_df)

def check_skew(top_products, data):
    newDict = {
        'Item': [],
        'Skewness Price':[],
        'Skewness Quantity':[]
    }
    for name in top_products:
        df = summary2(name, data)
        skew = df.skew()
        newDict['Item'].append(name)
        newDict['Skewness Price'].append(skew[0])
        newDict['Skewness Quantity'].append(skew[1])
        
    return pd.DataFrame(newDict)

def all_top_regress(i, data):
    testing = data.loc[data['name'] == i]
    tester = testing.groupby(['disc_price']).size()
    test_df = tester.to_frame().reset_index()
    test_df.columns = ['UnitPrice','Quantity']
    return display_regression_plot(test_df)

def ols_summary_log_skewQ(data):
    model = ols("np.log(Quantity) ~ UnitPrice", data = data).fit()
    st.text(model.summary())
    coefficient = model.params[1]
    r2 = model.rsquared
    partial_regression_plot(model, False)

def ols_summary_log_skewP(data):
    model = ols("Quantity ~ np.log(UnitPrice)", data = data).fit()
    st.text(model.summary())
    coefficient = model.params[1]
    r2 = model.rsquared
    partial_regression_plot(model, False)

def ols_summary_log_skewBoth(data):
    model = ols("np.log(Quantity) ~ np.log(UnitPrice)", data = data).fit()
    st.text(model.summary())
    coefficient = model.params[1]
    r2 = model.rsquared
    partial_regression_plot(model, False)

def all_top(i, data):
    testing = data.loc[data['name'] == i]
    tester = testing.groupby(['disc_price']).size()
    test_df = tester.to_frame().reset_index()
    test_df.columns = ['UnitPrice','Quantity']
    
    skew = []
    skew_values = test_df.skew()
    skew.append(skew_values[0])
    skew.append(skew_values[1])
    
    if abs(skew[0]) > 0.5 and abs(skew[1]) > 0.5:
        return ols_summary_log_skewBoth(test_df)
    
    elif abs(skew[0]) < 0.5 and abs(skew[1]) > 0.5:
        return ols_summary_log_skewQ(test_df)
    
    elif abs(skew[0]) > 0.5 and abs(skew[1]) < 0.5:
        return ols_summary_log_skewP(test_df)

def calculate_r2(product, name_map1, top_products2, data, final_df):
    if product not in top_products2:
        id = name_map1[product]
        data = data[data['StockCode'] == id]

        model = ols("np.log(Quantity) ~ np.log(UnitPrice)", data = data).fit()
        return model.rsquared

    else:
        testing = final_df.loc[final_df['name'] == product]
        tester = testing.groupby(['disc_price']).size()
        test_df = tester.to_frame().reset_index()
        test_df.columns = ['UnitPrice','Quantity']
        
        skew = []
        skew_values = test_df.skew()
        skew.append(skew_values[0])
        skew.append(skew_values[1])
        
        if abs(skew[0]) > 0.5 and abs(skew[1]) > 0.5:
            model = ols("np.log(Quantity) ~ np.log(UnitPrice)", data = test_df).fit()
            return model.rsquared
        
        elif abs(skew[0]) < 0.5 and abs(skew[1]) > 0.5:
            model = ols("np.log(Quantity) ~ UnitPrice", data = test_df).fit()
            return model.rsquared
        
        elif abs(skew[0]) > 0.5 and abs(skew[1]) < 0.5:
            model = ols("Quantity ~ np.log(UnitPrice)", data = data).fit()
            return model.rsquared
################################################################################################################################ Function Definitions ####################################################################################
def app():

    st.title('Regression Analysis')

    data = read_file('data.csv')

    top_products1 = ['23166', '85099B', '85123A', '23084', '22197']
    top_products2 = ['JBL Clip2 Portable Speaker',
                    'Yamaha - Natural Sound 5 2-Way All-Weather Outdoor Speakers (Pair) - White"', 
                    'Russound - Acclaim 5 Series 6-1/2 2-Way Indoor/Outdoor Speakers (Pair) - White"',
                    'MCR-B043 30W Bluetooth Wireless Music System (Black)',
                    'Kicker DSC44 4 D-Series 2-Way Car Speakers with 1/2" Tweeters"',
                    'Alpine - 6-1/2 2-Way Component Car Speakers with Poly-Mica Cones (Pair) - Black"',
                    'Details About Alpine 400w 5.25 Typee Coaxial 2way Car Speakers | Spe5000"']

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

    df = read_file('category_price.csv')

################################################################################################################################ Data Preprocessing for Household Products ####################################################################################
    data = data[(data['Quantity']> 0) & (data['UnitPrice'] > 0)]
    #Remove Outlier
    from scipy import stats
    data = data[(np.abs(stats.zscore(data['Quantity'])) < 3)]

    data = data.reindex(data.index.repeat(data.Quantity)) #multiply the quantity of items to expand the number of rows
    data['InvoiceDate']= pd.to_datetime(data['InvoiceDate']) #converting column invoice date to datetime format
    data = data.set_index('InvoiceDate') #setting date as an index for the dataframe

    #Adding additional time-based columns
    data['Year'] = data.index.year
    data['Month'] = data.index.month
    data['Weekday Name'] = data.index.day_name()
    data['Hour'] = data.index.hour
################################################################################################################################ Data Preprocessing for Electronic Products ####################################################################################
    df['Date_imp'] = pd.to_datetime(df['Date_imp'], format='%Y-%m-%d %H:%M:%S')
    df['hour'] = df['Date_imp'].dt.hour

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

    valid_items = df_detect[df_detect['disc_price'] > 5]
    valid_items = valid_items['name'].tolist()

    #Print Deleted Impressions with price variability
    df_clean = df_us[df_us['name'].isin(valid_items)]
    df_clean = df_clean[df_clean['name'].isin(top_products2)]


    frames = []
    for i in top_products2:
        test = df_clean[df_clean['name'] == i]
        test = test[(np.abs(stats.zscore(test['disc_price'])) < 3)]
        frames.append(test)

    final_df = pd.concat(frames)
################################################################################################################################ Description of Regression Models ####################################################################################

    st.header('Choice of Regression Models')

    st.subheader('Ordinary Least Squares Regression')
    st.text('''
    Ordinary Least Squares (OLS) regression is a statistical method of analysis that 
    estimates the relationship between one or more independent variables and a 
    dependent variable. It estimates the aforementioned relationship by minimizing 
    the sum of  squares in the difference between the observed and predicted values 
    of the dependent variable configured as a straight line.
    ''')

    st.subheader('Recursive Least Squares Regression')
    st.text('''
    Recursive least squares (RLS) is an expanding window version of ordinary least squares. 
    In addition to recursively calculating the regression coefficients, it aims to minimize
    a weighted linear least squares cost function rather than the usual mean square error.
    ''')

    st.subheader('Polynomial Regression')
    st.text('''
    As our straight line is unable to capture the salient pattern in the data, we increased
    the complexity of the model to overcome the under-fitting issue.
    ''')
################################################################################################################################ Displaying of Graphs ####################################################################################
    list_of_products = ['MEDIUM CERAMIC TOP STORAGE JAR',
                        'JUMBO BAG RED RETROSPOT',
                        'WHITE HANGING HEART T-LIGHT HOLDER',
                        'RABBIT NIGHT LIGHT',
                        'SMALL POPCORN HOLDER',
                        'JBL Clip2 Portable Speaker',
                        'Yamaha - Natural Sound 5 2-Way All-Weather Outdoor Speakers (Pair) - White"', 
                        'Russound - Acclaim 5 Series 6-1/2 2-Way Indoor/Outdoor Speakers (Pair) - White"',
                        'MCR-B043 30W Bluetooth Wireless Music System (Black)',
                        'Kicker DSC44 4 D-Series 2-Way Car Speakers with 1/2" Tweeters"',
                        'Alpine - 6-1/2 2-Way Component Car Speakers with Poly-Mica Cones (Pair) - Black"',
                        'Details About Alpine 400w 5.25 Typee Coaxial 2way Car Speakers | Spe5000"']

    product = st.selectbox('Pick a product', options = list_of_products, key = 1)

    if product not in top_products2:
        id = name_map1[product]
    
        st.subheader('Regression Plot')
        display_regression_plot(data[data['StockCode'] == id])

        st.subheader('Ordinary Least Squares (OLS) Estimation')
        ols_summary_log(data[data['StockCode'] == id],product)

        st.subheader('Rercursive Least Squares (RLS) Estimation')
        rls_summary_log(data[data['StockCode'] == id])

        st.subheader('Conclusion for '+product)
        st.text('''Both Ordinary Least Squares (OLS) & Recursive Least Squares (RLS) are consistent 
with the fact that price exhibits a OPPOSITE influence on quantity.
''')

    else:  
        st.subheader("Regression Plot")
        all_top_regress(product, final_df)

        st.subheader('Ordinary Least Squares (OLS) Estimation')
        all_top(product, final_df)

        st.subheader('Rercursive Least Squares (RLS) Estimation')
        testing = final_df.loc[final_df['name'] == i]
        tester = testing.groupby(['disc_price']).size()
        test_df = tester.to_frame().reset_index()
        test_df.columns = ['UnitPrice','Quantity']
        rls_summary_log(test_df)
        st.subheader('Conclusion for '+product)
        st.text('''Both Ordinary Least Squares (OLS) & Recursive Least Squares (RLS) are consistent 
with the fact that price exhibits a POSITIVE influence on quantity.
''')
        st.text('''
        Based on the P values above, we are unable to reject the null hypothesis of 2 items: Yamaha
        - Natural Sound 5 2-Way All-Weather Outdoor Speakers (Pair) - White'' and  Details About 
        Alpine 400w 5.25 Typee Coaxial 2way Car Speakers | Spe5000" as they  have a P value of 0.052
        and 0.085, above the significance level. Hence, there might not be a real relationship 
        between price and quantity even if the r^2 is high as it might be due to pure chance. 
        However, for the rest of the items, we can see that P Value is below the significance level 
        of 0.05 and can reject the null hypothesis. R^2 values for all the 7 products are at least 
        about 0.5 which represents that the  model is able to explain the variation in quantity 
        demanded at least half the time. This models will then be used for PED calculations.
        ''')

    newDict = {
        'Item': [],
        'R Squared Value':[]
    }

    for product in list_of_products:
        newDict['Item'].append(product)
        r2 = calculate_r2(product, name_map1, top_products2, data, final_df)
        newDict['R Squared Value'].append(r2)

    r2_df = pd.DataFrame(newDict)
    st.dataframe(r2_df)
    item = r2_df['Item']
    r2_score = r2_df['R Squared Value']

    n = len(list_of_products)
    source = ColumnDataSource(data=dict(item=item, r2_score=r2_score, color=cividis(n)))

    max_y = max(r2_score)

    p = figure(x_range=item, y_range=(0,max_y), plot_height=500, title="R Squared Value of each Product",
           toolbar_location=None, tools="")
    
    p.vbar(x='item', top='r2_score', width=0.9, color='color', source=source)
    
    p.xgrid.grid_line_color = None
    p.legend.orientation = "horizontal"
    p.legend.location = "top_center"
    p.xaxis.visible = False

    p.add_tools(HoverTool(tooltips=[("Product: ", "@item"),("R2 value:", "@r2_score")]))

    st.bokeh_chart(p, use_container_width=True)

    st.text('''
    While the R squared values are not perfect, the majority of products have low p values which 
    indicate a real relationship between price & quantity of products sold. The only caveat here
    is models with lower  R square have a larger prediction interval. This would result in 
    predictions that may not be as precise for a given regression equation. However, since we are
    not using the model to predict price or quantity for a given product. The R squared value is
    not taken with as much importance as we can already ascertain that the products' elasticity 
    through the changes in quantity & price.
''')

    st.text('''To further improve the R squared value of the household products, 
we will attempt to improve its complexity via polynomial regression in a separate section.''')