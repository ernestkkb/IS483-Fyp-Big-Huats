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

def summary(pid, data):
    df = data[data['StockCode'] == pid]
    df = df.reindex(df.index.repeat(df.Quantity))
    grouping = df.groupby(['UnitPrice']).size()
    grouping_df = grouping.to_frame().reset_index()
    grouping_df.columns = ['UnitPrice','Quantity']
    return pd.DataFrame(grouping_df)

def df_datepricequantity(data, stockID):
    date_price_qty = data[data['StockCode'] == stockID].groupby([data[data['StockCode'] == stockID].index, 'UnitPrice']).size()
    date_price_qty_df = date_price_qty.to_frame().reset_index()
    date_price_qty_df.columns = ['dates', 'price', 'quantity']
    return date_price_qty_df

def price_distribution_chart(data, stockID):
    data_selected = data[data['StockCode'] == stockID]
    
    fig = figure(
            title = "Price Distribution Chart",
             x_axis_label = 'Time',
             y_axis_label = 'Unit Price',
             width = 800,
             height = 400,
             x_axis_type='datetime'
            )

    fig.line(data_selected.index,data_selected['UnitPrice'],
             line_alpha = 0.8,
             line_width = 2
            )

    fig.add_tools(HoverTool(
        tooltips='<font face="Arial" size="3">Date: @x{%F}, Price: @y{0.00}</font>',
        mode='vline',
        formatters={'x': 'datetime'}
    ))

    st.bokeh_chart(fig, use_container_width=True)

def boxplot_month(data, stockID):
    df =data[data['StockCode'] == stockID]
    fig = go.Figure()

    fig.update_layout(
        # title="Distribution of price per month",
        xaxis_title="Months",
        yaxis_title="Unit Price",
        font=dict(
            family="Arial",
            size=12,
            color="#0E0D0C"
        )
    )
    fig.add_trace(go.Box(y=df['UnitPrice'], x=df['Month'])) 
    st.plotly_chart(fig, use_container_width=True)

def boxplot_day(data, stockID):
    order = {"Weekday Name": ['Monday', 'Tuesday', 'Wednesday','Thursday','Friday','Saturday','Sunday']}
    df =data[data['StockCode'] == stockID]
    fig = px.box(df, x="Weekday Name", y="UnitPrice",
            #  title="Distribution of price per day",
            category_orders=order)    
    st.plotly_chart(fig, use_container_width=True)

def rolling_mean(data, stockID):
    data_7d = data[data['StockCode'] == stockID]['UnitPrice'].rolling(7, center=True).mean()
    data_365d = data[data['StockCode'] == stockID]['UnitPrice'].rolling(window=365, center=True, min_periods=360).mean()
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(data[data['StockCode'] == stockID]['UnitPrice'], marker='.', markersize=2, color='0.6',linestyle='None', label='Daily')
    ax.plot(data_7d, linewidth=2, label='7-d Rolling Mean')
    ax.plot(data_365d, color='0.2', linewidth=3,label='Trend (365-d Rolling Mean)')
    
    ax.legend()
    ax.set_xlabel('Year')
    ax.set_ylabel('Unit Price')
    ax.set_title('Trends in Price Strategy');
    st.pyplot()

def df_hour_quantity(data, stockID):
    hr_qty = data[data['StockCode'] == stockID].groupby('Hour').size()
    hr_qty_df = hr_qty.to_frame().reset_index()
    hr_qty_df.columns = ['hour', 'quantity']
    return hr_qty_df

def bar_chart_hour(hr_qty_df):
    hour = hr_qty_df['hour']
    quantity = hr_qty_df['quantity']
    
    n= len(hour)
    
    source = ColumnDataSource(data=dict(hour=hour, quantity=quantity, color=inferno(n)))
    
    max_y = max(quantity)
    
    p = figure(x_range=(0,23), y_range=(0,max_y), plot_height=500, title="Number of items sold per hour",
           toolbar_location=None, tools="")
    
    p.vbar(x='hour', top='quantity', width=0.9, color = 'color', source=source)
    
    p.xgrid.grid_line_color = None
    p.legend.orientation = "horizontal"
    p.legend.location = "top_center"
    
    p.add_tools(HoverTool(tooltips=[("Hour","@hour"),("Quantity", "@quantity")]))

    st.bokeh_chart(p, use_container_width=True)

def df_date_quantity(data, stockID):
    date_qty = pd.DataFrame(data[data['StockCode'] == stockID].groupby(data[data['StockCode'] == stockID].index).size().reset_index())
    date_qty.columns = ['dates','quantity']
    return date_qty

def bar_chart_date(date_qty):
    date_qty['dates']= pd.to_datetime(date_qty['dates'])
    date = date_qty['dates']
    quantity = date_qty['quantity']
    n = len(date)
    
    source = ColumnDataSource(data=dict(date=date, quantity=quantity))
    
    max_y = max(quantity)
  
    p = figure(x_range= (min(date),max(date)),
               y_range=(0,max_y),
               plot_height=500,
               title="Date of highest purchase",
               x_axis_type = 'datetime',
               toolbar_location=None, 
               tools="")
    
    p.vbar(x='date', top='quantity', width=0.9, source=source)
    
    p.xgrid.grid_line_color = None
    p.legend.orientation = "horizontal"
    p.legend.location = "top_center"
    
    p.add_tools(HoverTool(
        tooltips='<font face="Arial" size="3">Date: @x{%F}, Price: @y{0}</font>',
        mode='vline',
        formatters={'date': 'datetime'}
    ))

    st.bokeh_chart(p, use_container_width=True)

def price_chart(data, stockID):
    percentiles = data[data['StockCode'] == stockID]['UnitPrice'].quantile([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, .9, 0.95, 1 ])
    
    fig = figure(title = "Price Percentile Distribution",
             x_axis_label = 'Percentile',
             y_axis_label = 'Price',
             width = 800,
             height = 400
            )

    fig.line(percentiles.index,percentiles,
             line_alpha = 0.8,
             line_width = 2
            )

    fig.add_tools(HoverTool(
        tooltips='<font face="Arial" size="3">Percentile: @x{0.00}, Price: @y{0.00}</font>',
        mode='vline'
    ))
    
    st.bokeh_chart(fig, use_container_width=True)

def quartile_barchart(data, stockID):
    data = data[data['StockCode'] == stockID]
    data['Quantile'] = pd.qcut(data['UnitPrice'], q=np.arange(0,1.1,0.1), duplicates='drop')
    df_tempo = pd.DataFrame(data.groupby('Quantile').agg('size').reset_index())
    df_tempo.columns = ['Quantile','Quantity']
    
    quantile = df_tempo['Quantile'].tolist()
    quantile = [str(i) for i in quantile]
    quantity = df_tempo['Quantity']
    n = len(quantile)
    
    max_y = max(quantity)
    source = ColumnDataSource(data=dict(quantile=quantile, quantity=quantity, color = cividis(n)))
    p = figure(x_range=quantile,y_range=(0,max_y), plot_height=500, title="Histogram by Price Percentiles",
           toolbar_location=None, tools="")
        
    p.vbar(x='quantile', top='quantity', width=0.9, color ='color', source=source)
    
    p.xgrid.grid_line_color = None
    p.legend.orientation = "horizontal"
    p.legend.location = "top_center"
    
    p.add_tools(HoverTool(tooltips=[("Quantity", "@quantity")]))

    st.bokeh_chart(p, use_container_width=True)

def day_df(data,stockID):
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Sunday"]
    day_qty = pd.DataFrame(data[data['StockCode'] == stockID].groupby('Weekday Name').size().reindex(order))
    day_qty.columns = ['quantity']
    return day_qty

def day_bar(data, stockID):
    day_qty = day_df(data,stockID)

    weekday = day_qty.index.tolist()
    quantity = day_qty['quantity']
    
    source = ColumnDataSource(data=dict(weekday=weekday, quantity=quantity, color=Spectral7))
    
    max_y = max(quantity)
    p = figure(x_range=weekday, y_range=(0,max_y), plot_height=500, title="Items sold per Day",
           toolbar_location=None, tools="")
    
    p.vbar(x='weekday', top='quantity', width=0.9, color='color', source=source)
    
    p.xgrid.grid_line_color = None
    p.legend.orientation = "horizontal"
    p.legend.location = "top_center"
    
    p.add_tools(HoverTool(tooltips=[("Quantity", "@quantity")]))

    st.bokeh_chart(p, use_container_width=True)
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
    list_of_products = ['MEDIUM CERAMIC TOP STORAGE JAR',
                        'JUMBO BAG RED RETROSPOT',
                        'WHITE HANGING HEART T-LIGHT HOLDER',
                        'RABBIT NIGHT LIGHT',
                        'SMALL POPCORN HOLDER']
                        # 'JBL Clip2 Portable Speaker',
                        # 'Yamaha - Natural Sound 5 2-Way All-Weather Outdoor Speakers (Pair) - White"', 
                        # 'Russound - Acclaim 5 Series 6-1/2 2-Way Indoor/Outdoor Speakers (Pair) - White"',
                        # 'MCR-B043 30W Bluetooth Wireless Music System (Black)',
                        # 'Kicker DSC44 4 D-Series 2-Way Car Speakers with 1/2" Tweeters"',
                        # 'Alpine - 6-1/2 2-Way Component Car Speakers with Poly-Mica Cones (Pair) - Black"',
                        # 'Details About Alpine 400w 5.25 Typee Coaxial 2way Car Speakers | Spe5000"']

    product = st.selectbox('Pick a product', options = list_of_products, key = 1)

    id = name_map1[product]

    st.header("Household Product: "+product)
    st.subheader('Price Percentile Distribution')
    price_chart(data,id)

    st.subheader('Histogram by Price Percentiles')
    quartile_barchart(data, id)

    st.subheader('Quantity of Items Sold per Day')
    st.write(day_df(data,id)) 
    # st.dataframe(day_df(data,id))

    st.subheader('Bar Chart of Items Sold per Day')
    day_bar(data, id)

    st.subheader('Price Distribution per Date & Time')
    st.write(df_datepricequantity(data,id)) 
    # st.dataframe(df_datepricequantity(data,id))
    
    st.subheader('Price Distribution Chart')
    price_distribution_chart(data, id)

    st.subheader('Distribution of price per month')
    boxplot_month(data, id)

    st.subheader('Distribution of price per day')
    boxplot_day(data, id)

    st.subheader('Number of items sold per hour')
    st.write(df_hour_quantity(data,id)) 
    # st.dataframe(df_hour_quantity(data, id))
    bar_chart_hour(df_hour_quantity(data, id))

    st.subheader('Quantity of items Sold per Date & Time')
    st.write(df_date_quantity(data, id).sort_values(by=['quantity'])) 
    # st.dataframe(df_date_quantity(data, id).sort_values(by=['quantity']))
    bar_chart_date(df_date_quantity(data, id))


