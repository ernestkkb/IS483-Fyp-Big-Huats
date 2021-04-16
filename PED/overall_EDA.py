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
    if(filename == 'data.csv'):
        data = pd.read_csv(filename, encoding = 'ISO-8859-1')
    else:
        data = pd.read_csv(filename)
    return data

def df_hour_quantity(data, stockID):
    hr_qty = data[data['StockCode'] == stockID].groupby('Hour').size()
    hr_qty_df = hr_qty.to_frame().reset_index()
    hr_qty_df.columns = ['hour', 'quantity']
    return hr_qty_df

def day_df(data,stockID):
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Sunday"]
    day_qty = pd.DataFrame(data[data['StockCode'] == stockID].groupby('Weekday Name').size().reindex(order))
    day_qty.columns = ['quantity']
    return day_qty

def summary(pid, data):
    df = data[data['StockCode'] == pid]
    df = df.reindex(df.index.repeat(df.Quantity))
    grouping = df.groupby(['UnitPrice']).size()
    grouping_df = grouping.to_frame().reset_index()
    grouping_df.columns = ['UnitPrice','Quantity']
    return pd.DataFrame(grouping_df)

def summary2(name, data):
    df = data[data['name'] == name]
    grouping = df.groupby(['disc_price']).size()
    grouping_df = grouping.to_frame().reset_index()
    grouping_df.columns = ['UnitPrice','Quantity']
    return pd.DataFrame(grouping_df)

def price_distribution_chart_overall(top_products, data, name_map = None):
    
    layout = go.Layout(
        autosize=False,
        width= 1000,
        height= 500
    )

    fig = go.Figure(layout = layout)

    if(name_map):
        for name in top_products:
            name_df = summary(name,data)
            fig.add_trace(go.Scatter(y=name_df['Quantity'], x= name_df['UnitPrice'], mode = 'lines', name = name_map[name]))
        st.plotly_chart(fig, use_container_width=True)

    else:
        for name in top_products:
            name_df = summary2(name, data)
            fig.add_trace(go.Scatter(y=name_df['Quantity'], x= name_df['UnitPrice'], mode = 'lines', name = name))
        st.plotly_chart(fig, use_container_width=True)


def day_df2(data,name):
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_qty = pd.DataFrame(data[data['name'] == name].groupby('Day_n').size().reindex(order))
    day_qty.columns = ['quantity']
    return day_qty

def day_bar_overall(top_products, data, name_map = None):
    update = []

    layout = go.Layout(
        autosize=False,
        width= 1000,
        height= 500
    )

    if(name_map):
        for name in top_products:
            day_qty = day_df(data,name)
            update.append(go.Bar(name = name_map[name], x=day_qty.index, y= day_qty['quantity']))
        
        fig = go.Figure(update, layout = layout)
        fig.update_layout(barmode = 'stack')
        st.plotly_chart(fig, use_container_width=True)
    else:
        for name in top_products:
            day_qty = day_df2(data,name)
            update.append(go.Bar(name = name, x=day_qty.index, y= day_qty['quantity']))
        
        fig = go.Figure(update, layout = layout)
        fig.update_layout(barmode = 'stack')
        st.plotly_chart(fig, use_container_width=True)


def df_hour_quantity2(data, name):
    hr_qty = data[data['name'] == name].groupby('hour').size()
    hr_qty_df = hr_qty.to_frame().reset_index()
    hr_qty_df.columns = ['hour', 'quantity']
    return hr_qty_df

def bar_chart_hour_overall(top_products, data, name_map = None):
    update = []

    layout = go.Layout(
        autosize=False,
        width= 1000,
        height= 500
    )

    if(name_map):
        for name in top_products:
            hr_qty_df = df_hour_quantity(data, name)
            update.append(go.Bar(name = name_map[name], x=hr_qty_df['hour'], y= hr_qty_df['quantity']))
        
        fig = go.Figure(update, layout = layout)
        fig.update_layout(barmode = 'stack')
        st.plotly_chart(fig, use_container_width=True)
    else:
        for name in top_products:
            hr_qty_df = df_hour_quantity2(data, name)
            update.append(go.Bar(name = name, x=hr_qty_df['hour'], y= hr_qty_df['quantity']))
        
        fig = go.Figure(update, layout = layout)
        fig.update_layout(barmode = 'stack')
        st.plotly_chart(fig, use_container_width=True)


def df_month_quantity2(data, name):
    month_qty = pd.DataFrame(data[data['name'] == name].groupby('month').size()).reset_index()
    month_qty.columns = ['month','quantity']
    return month_qty


def df_month_quantity(data, stockID):
    mth_qty = data[data['StockCode'] == stockID].groupby('Month').size()
    mth_qty_df = mth_qty.to_frame().reset_index()
    mth_qty_df.columns = ['month', 'quantity']
    return mth_qty_df

def bar_chart_month_overall(top_products, data, name_map = None):
    update = []
    layout = go.Layout(
        autosize=False,
        width= 1000,
        height= 500
    )

    if(name_map):
        for name in top_products:
            month_qty_df = df_month_quantity(data, name)
            update.append(go.Bar(name = name_map[name], x=month_qty_df['month'], y= month_qty_df['quantity']))
        
        fig = go.Figure(update, layout = layout)
        fig.update_layout(barmode = 'stack')
        st.plotly_chart(fig, use_container_width=True)

    else:
        for name in top_products:
            month_qty_df = df_month_quantity2(data, name)
            update.append(go.Bar(name = name, x=month_qty_df['month'], y= month_qty_df['quantity']))
        
        fig = go.Figure(update, layout = layout)
        fig.update_layout(barmode = 'stack')
        st.plotly_chart(fig, use_container_width=True)
################################################################################################################################ Function Definitions ####################################################################################
#Reading both files

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

    df = read_file('category_price.csv')
################################################################################################################################ Data Preprocessing for Household Products ####################################################################################
    #Remove Outlier
    from scipy import stats
    data = data[(np.abs(stats.zscore(data['Quantity'])) < 3)]

    data = data[(data['Quantity']> 0) & (data['UnitPrice'] > 0)]
    data1 = data

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
################################################################################################################################ Displaying of graphs ####################################################################################
    product = st.selectbox('Pick a product category', options=['Household', 'Electronic'], key = 2)
    if(product == "Household"):
        st.subheader('Price Distribution Chart')
        price_distribution_chart_overall(top_products1, data1, name_map = name_map)
        st.subheader('Day of week of highest purchase')
        day_bar_overall(top_products1, data, name_map = name_map)
        st.subheader('Hour of highest purchase')
        bar_chart_hour_overall(top_products1, data, name_map = name_map)
        st.subheader('Month of highest purchase')
        bar_chart_month_overall(top_products1, data, name_map = name_map)

    else:
        st.subheader('Price Distribution Chart')
        price_distribution_chart_overall(top_products2, final_df)
        st.subheader('Day of week of highest purchase')
        day_bar_overall(top_products2, final_df)
        st.subheader('Hour of highest purchase')
        bar_chart_hour_overall(top_products2, final_df)
        st.subheader('Month of highest purchase')
        bar_chart_month_overall(top_products2, final_df)