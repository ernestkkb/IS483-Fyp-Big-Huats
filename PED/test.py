import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from IPython.display import display, HTML
from bokeh.plotting import figure, output_notebook, show
from bokeh.models.tools import HoverTool
from bokeh.io import output_notebook
from bokeh.models import ColumnDataSource
from bokeh.palettes import Spectral7,Category20c
from bokeh.plotting import figure
from bokeh.transform import cumsum
import streamlit as st
import plotly.graph_objects as go

@st.cache()
def read_file (filename):
    data = pd.read_csv(filename, encoding = 'ISO-8859-1')
    return data

data = read_file('data.csv')
# Remove noise data from quantity & UnitPrice 
data = data[(data['Quantity']> 0) & (data['UnitPrice'] > 0)]
data = data.reindex(data.index.repeat(data.Quantity)) #multiply the quantity of items to expand the number of rows
data['InvoiceDate']= pd.to_datetime(data['InvoiceDate']) #converting column invoicedate to datetime format
data = data.set_index('InvoiceDate') #setting date as an index for the dataframe
#Adding additional time-based columns
data['Year'] = data.index.year
data['Month'] = data.index.month
data['Weekday Name'] = data.index.day_name()
data['Hour'] = data.index.hour

#Remove Outlier
from scipy import stats
data = data[(np.abs(stats.zscore(data['Quantity'])) < 3)]

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

chosen = ['23166', '85099B', '85123A', '23084', '22197']
price_distribution_chart(data, chosen[0])

def boxplot_month(data, stockID):
    df =data[data['StockCode'] == stockID]
    fig = go.Figure()

    fig.update_layout(
        title="Distribution of price per month",
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

boxplot_month(data, chosen[0])