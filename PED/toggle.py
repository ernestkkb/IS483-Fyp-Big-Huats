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
@st.cache()
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

def summary2(name, data):
    df = data[data['name'] == name]
    grouping = df.groupby(['disc_price']).size()
    grouping_df = grouping.to_frame().reset_index()
    grouping_df.columns = ['UnitPrice','Quantity']
    return pd.DataFrame(grouping_df)

def price_distribution_chart_overall(top_products, name_map = None):
    fig = go.Figure()
    
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


def day_bar_overall(top_products, name_map = None):
    update = []

    if(name_map):
        for name in top_products:
            day_qty = day_df(data,name)
            update.append(go.Bar(name = name_map[name], x=day_qty.index, y= day_qty['quantity']))
        
        fig = go.Figure(update)
        fig.update_layout(barmode = 'stack')
        fig.show()
    else:
        for name in top_products:
            day_qty = day_df2(final_df,name)
            update.append(go.Bar(name = name, x=day_qty.index, y= day_qty['quantity']))
        
        fig = go.Figure(update)
        fig.update_layout(barmode = 'stack')
        st.plotly_chart(fig, use_container_width=True)


def df_hour_quantity2(data, name):
    hr_qty = data[data['name'] == name].groupby('hour').size()
    hr_qty_df = hr_qty.to_frame().reset_index()
    hr_qty_df.columns = ['hour', 'quantity']
    return hr_qty_df

    
def bar_chart_hour_overall(top_products, name_map = None):
    update = []

    if(name_map):
        for name in top_products:
            hr_qty_df = df_hour_quantity(data, name)
            update.append(go.Bar(name = name_map[name], x=hr_qty_df['hour'], y= hr_qty_df['quantity']))
        
        fig = go.Figure(update)
        fig.update_layout(barmode = 'stack')
        st.plotly_chart(fig, use_container_width=True)
    else:
        for name in top_products:
            hr_qty_df = df_hour_quantity2(final_df, name)
            update.append(go.Bar(name = name, x=hr_qty_df['hour'], y= hr_qty_df['quantity']))
        
        fig = go.Figure(update)
        fig.update_layout(barmode = 'stack')
        st.plotly_chart(fig, use_container_width=True)
        st.text('''Sales remain high even in the wee hours, indicating people might be 
    increasingly scrolling through sites and making purchases at late hours. This supports a 
    study by John Lewis Partnership Card and released by BBC that "More consumers seem to be 
    shopping online late at night and in the early hours of the morning, say retailers. New 
    data from the John Lewis Partnership Card shows that one in 15 purchases are now made 
    between the hours of midnight and 06:00. The research shows that the number of purchases 
    made in this period rose by 23% in 2018, compared with 2017." Apart from this finding, 
    we find that 3pm and 5pm have higher levels of purchase too for most products''')
    
def df_month_quantity(data, name):
    month_qty = pd.DataFrame(data[data['name'] == name].groupby('month').size()).reset_index()
    month_qty.columns = ['month','quantity']
    return month_qty

def bar_chart_month_overall(top_products, data, name_map = None):
    update = []

    if(name_map):
        for name in top_products:
            month_qty_df = df_month_quantity(data, name)
            update.append(go.Bar(name = name_map[name], x=month_qty_df['month'], y= month_qty_df['quantity']))
        
        fig = go.Figure(update)
        fig.update_layout(barmode = 'stack')
        st.plotly_chart(fig, use_container_width=True)

    else:
        for name in top_products:
            month_qty_df = df_month_quantity(data, name)
            update.append(go.Bar(name = name, x=month_qty_df['month'], y= month_qty_df['quantity']))
        
        fig = go.Figure(update)
        fig.update_layout(barmode = 'stack')
        st.plotly_chart(fig, use_container_width=True)
        st.text('''Based on this data from March till December, we can see that there are generally 
    higher number of purchases in months July and August for each item. We can attribute this 
    to the holiday season coming to an end, where both students & working adults get back to 
    school / work and manufacturers are found to generally promote products during this 
    back-to-school/work period''')
################################################################################################################################ Function Definitions ####################################################################################
#Reading both files 
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
st.subheader('Price Distribution Chart')
product = st.selectbox('Pick a product category',options=['Household', 'Electronic'])
if(product == "Household"):
    price_distribution_chart_overall(top_products1, name_map = name_map)
else:
    price_distribution_chart_overall(top_products2)

st.subheader('Day of week of highest purchase')
product = st.selectbox('Pick a product category', options=['Household', 'Electronic'])
if(product == "Household"):
    day_bar_overall(top_products1, name_map = name_map)
else:
    day_bar_overall(top_products2)

st.subheader('Hour of highest purchase')
product = st.selectbox('Pick a product category', options=['Household', 'Electronic'])
if(product == "Household"):
    bar_chart_hour_overall(top_products1, name_map = name_map)
else:
    bar_chart_hour_overall(top_products2)

st.subheader('Month of highest purchase')
product = st.selectbox('Pick a product category', options=['Household', 'Electronic'])
if(product == "Household"):
    bar_chart_month_overall(top_products1, data, name_map = name_map)
else:
    bar_chart_month_overall(top_products2, final_df)