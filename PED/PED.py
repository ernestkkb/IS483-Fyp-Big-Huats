import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.pyplot import figure
from IPython.display import display, HTML
import streamlit as st
from scipy import stats

import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff

from statsmodels.compat import lzip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols

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

def summary2(name, data):
        df = data[data['name'] == name]
        grouping = df.groupby(['disc_price']).size()
        grouping_df = grouping.to_frame().reset_index()
        grouping_df.columns = ['UnitPrice','Quantity']
        return pd.DataFrame(grouping_df)

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

def PED2(top_products, final_df): 
    newDict = {
        'Item': [],
        'Price Elasticity of Demand':[]
    }
    
    for name in top_products: 
        data = summary2(name, final_df)
        model = ols("np.log(Quantity) ~ UnitPrice", data = data).fit()
        intercept, slope = model.params
        mean_price = np.mean(data['UnitPrice'])
        mean_quantity = np.mean(data['Quantity'])
        price_elasticity = ((slope) * (mean_price/mean_quantity))

        newDict['Item'].append(name)
        newDict['Price Elasticity of Demand'].append(price_elasticity)
    
    return pd.DataFrame(newDict)
################################################################################################################################ Function Definitions ####################################################################################
def app():

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

    st.subheader('Price Elasticity of Demand')
    product = st.selectbox('Pick a product category', options=['Household', 'Electronic'], key = 2)
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
################################################################################################################################ Displaying of graphs ####################################################################################
    if(product == 'Household'):
        ped_df = PED(data)
        st.dataframe(ped_df)

        ped = ped_df['Price Elasticity of Demand']
        item = ped_df['Item']

        source = ColumnDataSource(data=dict(item=item, ped=ped, color=Spectral5))

        max_y = max(ped)

        p = figure(x_range=item, y_range=(0,max_y), plot_height=500, title="PED Value of each Product",
            toolbar_location=None, tools="")
        
        p.vbar(x='item', top='ped', width=0.9, color='color', source=source)
        
        p.xgrid.grid_line_color = None
        p.legend.orientation = "horizontal"
        p.legend.location = "top_center"
        p.xaxis.visible = False

        p.add_tools(HoverTool(tooltips=[("Product: ", "@item"),("PED value:", "@ped")]))

        st.bokeh_chart(p, use_container_width=True)

        st.subheader('Conclusion for all household products''')
        st.text('''
        Household items sold on ecommerce platforms tend to be inelastic where changes in price
        have a less than proportional effect on the quantity of the good demanded. This can be
        seen from the aforementioned 5 household items where we concluded that price had a
        minimal change in quantity of products sold. This means that consumers are not sensitive
        towards the changes in price (they buy about the same quantity even when prices change). 
        Furthermore, the relationship between price and quantity of household products is negative
        ''')

    else:
        ped_df = PED2(top_products2, final_df)
        st.dataframe(ped_df)

        ped = ped_df['Price Elasticity of Demand']
        item = ped_df['Item']

        source = ColumnDataSource(data=dict(item=item, ped=ped, color=Spectral7))

        max_y = max(ped)

        p = figure(x_range=item, y_range=(0,max_y), plot_height=500, title="PED Value of each Product",
            toolbar_location=None, tools="")
        
        p.vbar(x='item', top='ped', width=0.9, color='color', source=source)
        
        p.xgrid.grid_line_color = None
        p.legend.orientation = "horizontal"
        p.legend.location = "top_center"
        p.xaxis.visible = False

        p.add_tools(HoverTool(tooltips=[("Product: ", "@item"),("PED value:", "@ped")]))

        st.bokeh_chart(p, use_container_width=True)
        st.subheader('Conclusion for all electronic products')
        st.text('''
        We can see that there is quite a distribution of elasticities. Also, it is worthy to note
        that most of these Speaker products are considered Veblen goods (due to the positive PED)
        except for one. This means that as price increases, quantity demanded increases too. 
        Interpreting each item PED values are as follows:

        1) JBL Clip2 Portable speaker: PED of 0.461 indicates that a 10% increase in the price for
        the jar results in a 4.61% increase in quantity demanded. Absolute value of PED is less than 
        1 so its quantity demanded is price inelastic.

        2) Yamaha - Natural Sound 5 2-Way All-Weather Outdoor Speakers (Pair) - White: PED of 1.205
        indicates that a 10% increase in the price for the bag results in a 12% increase in quantity 
        demanded. Absolute value of PED is more than 1 so its quantity demanded is price elastic.

        3) Russound - Acclaim 5 Series 6-1/2 2-Way Indoor/Outdoor Speakers (Pair) - White": PED of 
        0.527 indicates that a 10% increase in the price for the light holder results in a 5.27% 
        increase in quantity demanded. Absolute value of PED is less than 1 so its quantity demanded 
        is price inelastic.

        4) MCR-B043 30W Bluetooth Wireless Music System (Black): PED of 2.58 indicates that a 10% 
        increase in the price for the light results in a 25% increase in quantity demanded. 
        Absolute value of PED is more than 1 so its quantity demanded is price elastic.

        5) Kicker DSC44 4 D-Series 2-Way Car Speakers with 1/2" Tweeters": PED of -1.619 indicates
        that a 10% increase in the price for the popcorn holder results in a 16% decrease in quantity
        demanded. Absolute value of PED is more than 1 so its quantity demanded is price elastic.

        6) Alpine - 6-1/2 2-Way Component Car Speakers with Poly-Mica Cones (Pair) - Black": PED of 
        0.675 indicates that a 10% increase in the price for the light holder results in a 6.75% 
        increase in quantity demanded. Absolute value of PED is less than 1 so its quantity demanded
        is price inelastic.

        7) Details About Alpine 400w 5.25 Typee Coaxial 2way Car Speakers | Spe5000": PED of 0.735 
        indicates that a 10% increase in the price for the light holder results in a 7.35% increase 
        in quantity demanded. Absolute value of PED is less than 1 so its quantity demanded is price inelastic.
        ''')
