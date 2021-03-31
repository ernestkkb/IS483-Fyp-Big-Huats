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

#Function Definitions

@st.cache()
def read_file (filename):
    data = pd.read_csv(filename, encoding = 'ISO-8859-1')
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

@st.cache(suppress_st_warning=True)
def app():
    st.title('Exploratory Data Analysis')

    st.header('Household products')
    data = read_file('data.csv')

    st.subheader('Preview of Data')
    st.dataframe(data.head())

    st.subheader('Summary Statistics of Data')
    st.dataframe(data.describe())

    # Remove noise data from quantity & UnitPrice 
    data = data[(data['Quantity']> 0) & (data['UnitPrice'] > 0)]

    st.subheader('Number of unique prices per product')
    chosen = ['23166', '85099B', '85123A', '23084', '22197']

    for pid in chosen:
        st.text(data[data['StockCode'] == pid].Description.iloc[0])
        st.dataframe(summary(pid,data))

    st.subheader('Data preprocessing')
    with st.echo():
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

    loop_dict = {
        'Medium Ceramic Storage Jar': ['''
        Over 90% of the transactions have prices that are at the prices of 1.04 - 1.25. 
        Only a small percentage range from $1.25 to 2.46.
        ''','The weekdays from Monday - Thursday is having better receptivity as compared to the weekends.', '''
        One common similarity between each month is that the prices would vary drastically from the 
        minimum (1.04) to the maximum (2.46) with the exception of September where the bulk of the 
        time, the prices remained low. 
        ''','The prices across the months all have a relatively constant spread across the year, with the exception of August & October.','The weekdays tend to see a more consistent spread of prices as compared to the weekend.','No apparent pricing strategy however one interesting observation is that prices tend to be increasing at a steady pace from August towards December.','10am and 2pm are popular timings for people to buy this particular item online.','The mid of may & the end of july are popular times of the year where increasing demand is seen.'],

        'Jumbo Bag Red Retrospot': ['''60% of the transactions have prices ranging from 1.65 to 1.79 
    while a very small group (8-9%) ranges from 2.08 to 4.95''','''The demand of the item increases from Monday to Thursday before slowly decreasing throughout the weekend. 
    Most items were bought up to Thursday, with Thursday being the peak. ''','''One common similarity between each month is that the prices would vary drastically 
    with the exception of May where the bulk of the time, the prices remained low & stable. ''','''The month of April has the most spread of prices compared to the other months. 
    Additionally, the median prices for months April to December are higher than the first 3 months of the year & December. 
    This could indicate a form of sale during the starting & ending of each year.''','The days, Wednesday & Thursday have the least amount of spread of prices.','''No apparent pricing strategy however one interesting observation is that 
    the prices of the item are seen to spiked around April while remaining relatively stable across the year.''','A large quantity of items are sold at between the timings of 10am to 12pm','''The mid of march, october & november are popular times of the year where increasing demand is seen. 
    This could be attributed to the fact that there might be holidays approaching.'''],

        'White Hanging Heart T-Light Holder' : ['''The majority of transactions are at the range of 2.399 - 2.55. 
    The next 2nd largest group of transactions have prices ranging from 2.55 to 2.95. 
    The transactions with prices between 2.95 - 3.2 forms the minority.''','The demand of the items are higher between Monday to Thursday before slowly decreasing from friday onwards.','''One common similarity between each month is that the prices would vary drastically 
    with the exception of september where the bulk of the time, the prices remained low. ''','The months April & November have the highest spread of prices compared to the other months.','The days, Monday & Thursday have the highest spread of prices.','''No apparent pricing strategy however one interesting observation is that the 
    prices of the item will spike every 2 months starting from June.''','A large quantity of items are sold at between the timings of 12pm to 1pm','The mid of January, April & November are popular times of the year where increasing demand is seen.'],

        'Rabbit Night Light': ['''60% of the transactions have prices ranging from 1.67 to 1.79 
    while a very small group (2%-3%) ranges from 4.13 to 4.96''', '''The demand of the items are higher between Tuesday to Thursday before slowly decreasing from friday onwards.''', 'The months from July leading up to October have less voltile prices.','The month September have the greatest spread of prices compared to the other months.','Monday has the greatest amount of spread of prices.','''No apparent pricing strategy however one interesting observation is that prices tend to be lower at the first half of the year (until November). 
    From November, the prices of the item are seen to be increasing at a steady pace and maintained throughout the last end of the year.''','A large quantity of items are sold at 12pm & 3pm.','The months between November to December are popular times of the year where increasing demand is seen.'],

        'Small Popcorn Holder':['80% of the transactions have prices ranging from the price of 0.72 to 0.85','The amount of quantity sold slowly increases throughout the week before dropping during the weekend','''One common similarity between each month is that the 
    prices would vary drastically with months from october having higher spikes''', 'The months March & November have the greatest spread of prices compared to the other months.','Monday has the greatest amount of spread of prices.','''No apparent pricing strategy however from October, 
    the prices of the item are seen to be increasing at a steady pace and maintained throughout the last end of the year.''','A large quantity of items are sold between 10am - 12pm','The mid of may has the highest amount of sales followed by the end of year (from November onwards)']

    }

    chosen = {
        '23166':'Medium Ceramic Storage Jar',
        '85099B':'Jumbo Bag Red Retrospot',
        '85123A':'White Hanging Heart T-Light Holder',
        '23084':'Rabbit Night Light',
        '22197':'Small Popcorn Holder'
    }

    for id in chosen:
        product_title = chosen[id]
        product_descriptions = loop_dict[product_title]

        st.header("Household Product: "+product_title)
        st.subheader('Price Percentile Distribution')
        price_chart(data,id)
        st.text(product_descriptions[0])
        st.subheader('Histogram by Price Percentiles')
        quartile_barchart(data, id)
        st.subheader('Quantity of Items Sold per Day')
        st.dataframe(day_df(data,id))
        st.subheader('Bar Chart of Items Sold per Day')
        day_bar(data, id)
        st.text(product_descriptions[1])
        st.subheader('Price Distribution per Date & Time')
        st.dataframe(df_datepricequantity(data,id))
        st.subheader('Price Distribution Chart')
        price_distribution_chart(data, id)
        st.text(product_descriptions[2])
        st.subheader('Distribution of price per month')
        boxplot_month(data, id)
        st.text(product_descriptions[3])
        st.subheader('Distribution of price per day')
        boxplot_day(data, id)
        st.text(product_descriptions[4])
        # st.subheader('Trends in Pricing Strategy')
        # rolling_mean(data, id)
        st.text(product_descriptions[5])
        st.subheader('Number of items sold per hour')
        st.dataframe(df_hour_quantity(data, id))
        bar_chart_hour(df_hour_quantity(data, id))
        st.text(product_descriptions[6])
        st.subheader('Quantity of items Sold per Date & Time')
        st.dataframe(df_date_quantity(data, id).sort_values(by=['quantity']))
        bar_chart_date(df_date_quantity(data, id))
        st.text(product_descriptions[7])

    st.header('General Conclusion for Household Products')
    st.subheader('Optimal Timings')
    st.text('''It has been observed that there is increasing demand of items bought between 10am - 12pm 
and occasionally early afternoons at 2-3pm.''')
    st.subheader('Pricing Strategy')
    st.text('''The second half of the year (from September/October) is often a good period to steadily 
increase your prices while keeping the quantity of items bought high.''')
    st.subheader('Weekdays > Weekends for Household Products')
    st.text('''It is better to have any kind of sale from Monday - Thursday as it has shown that these 
days have better receptivity compared to Friday - Sunday (Weekend)''')

    
    def read_file2(filename):
        data = pd.read_csv(filename)
        return data

    st.header('Electronic products')
    df = read_file2('category_price.csv')
    st.subheader('Preview of Data')
    st.dataframe(df.head())
    st.subheader('Data Preprocessing')


    with st.echo():
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
        df_detect

        valid_items = df_detect[df_detect['disc_price'] > 5]
        valid_items = valid_items['name'].tolist()

        #Print Deleted Impressions with price variability
        df_clean = df_us[df_us['name'].isin(valid_items)]

        #Print Deleted Impressions with price variability
        st.text("Sustainable Price Variability Impressions for further price regression analysis model: " + str(len(df_clean)))
        st.text("Percentage Deleted: "+"{:.2%}".format(1 - len(df_clean)/ len(df_us)))

    st.subheader('Price Boxplot per Category')
    ranks = df_clean.groupby("Category_name")["disc_price"].mean().fillna(0).sort_values()[::-1].index
    order = {"Category_name": ranks}
    fig = px.box(df, x="disc_price", y="Category_name",
                title="Price distribution per Category", category_orders=order, width = 3000, height = 1500)    
    st.plotly_chart(fig, use_container_width=True)


    #Detect relevant items for analysis
    top_products = ['JBL Clip2 Portable Speaker',
                'Yamaha - Natural Sound 5 2-Way All-Weather Outdoor Speakers (Pair) - White"', 
                'Russound - Acclaim 5 Series 6-1/2 2-Way Indoor/Outdoor Speakers (Pair) - White"',
                'MCR-B043 30W Bluetooth Wireless Music System (Black)',
                'Kicker DSC44 4 D-Series 2-Way Car Speakers with 1/2" Tweeters"',
                'Alpine - 6-1/2 2-Way Component Car Speakers with Poly-Mica Cones (Pair) - Black"',
                'Details About Alpine 400w 5.25 Typee Coaxial 2way Car Speakers | Spe5000"']

    df_clean = df_clean[df_clean['name'].isin(top_products)]
    st.dataframe(df_clean)

    st.subheader('Dropping Price Outliers')
    from scipy import stats
    with st.echo():
        frames = []
        for i in top_products:
            test = df_clean[df_clean['name'] == i]
            test = test[(np.abs(stats.zscore(test['disc_price'])) < 3)]
            frames.append(test)

        final_df = pd.concat(frames)
        st.dataframe(final_df)

    st.subheader('Price Distribution Plot for top selling products')
    hist_data_combined = []
    group_labels_combined = []
    colors = ['#A56CC1', '#A6ACEC', '#63F5EF','#F8B195','#F67280','#C06C84', '#6C5B7B']

    for name, selection in df_clean.groupby('name'):
        hist_data_combined.append(df_clean.loc[df_clean.name == name, "disc_price"])
        group_labels_combined.append(name)
        
        hist_data = [df_clean.loc[df_clean.name == name, "disc_price"]]
        group_labels = [name]
        
        fig = ff.create_distplot(hist_data, group_labels, show_rug=False)
        fig.update_layout(title_text= name)
        st.plotly_chart(fig, use_container_width=True)

    #Putting all together
    fig = ff.create_distplot(hist_data_combined, group_labels_combined, colors=colors,
                        bin_size=3.0, show_rug=False)

    # Add title
    fig.update_layout(title_text='Price Distribution Plot for top selling products')
    st.plotly_chart(fig, use_container_width=True)

    hist_data_combined = []
    group_labels_combined = []
    colors = ['#A56CC1', '#A6ACEC', '#63F5EF','#F8B195','#F67280','#C06C84', '#6C5B7B']

    for name, selection in final_df.groupby('name'):
        hist_data_combined.append(final_df.loc[final_df.name == name, "disc_price"])
        group_labels_combined.append(name)
        
        hist_data = [final_df.loc[final_df.name == name, "disc_price"]]
        group_labels = [name]
        
        fig = ff.create_distplot(hist_data, group_labels, show_rug=False)
        fig.update_layout(title_text= name)
        st.plotly_chart(fig, use_container_width=True)

    #Putting all together
    fig = ff.create_distplot(hist_data_combined, group_labels_combined, colors=colors,
                        bin_size=3.0, show_rug=False)

    # Add title
    fig.update_layout(title_text='Price Distribution Plot for top selling products')
    st.plotly_chart(fig, use_container_width=True)

    #Product Summaries
    def summary2(name):
        df = final_df[final_df['name'] == name]
        grouping = df.groupby(['disc_price']).size()
        grouping_df = grouping.to_frame().reset_index()
        grouping_df.columns = ['UnitPrice','Quantity']
        return pd.DataFrame(grouping_df)

    st.subheader('Number of unique prices per product')
    for name in top_products:
        st.text(name)
        st.dataframe(summary2(name))

    color = ['b-', 'g-', 'r-', 'y-', 'm-', 'c-', 'k-']
    st.subheader('Price Distribution Plots')
    def price_distribution_chart_overall(top_products):
        fig = go.Figure()
        
        for name in top_products:
            name_df = summary2(name)
            fig.add_trace(go.Scatter(y=name_df['Quantity'], x= name_df['UnitPrice'], mode = 'lines', name = name))
        st.plotly_chart(fig, use_container_width=True)

    price_distribution_chart_overall(top_products)

    st.subheader('Effect of Day of Week on Quantity Sold')
    def day_df2(data,name):
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_qty = pd.DataFrame(data[data['name'] == name].groupby('Day_n').size().reindex(order))
        day_qty.columns = ['quantity']
        return day_qty


    def day_bar_overall(top_products):
        data = []
        for name in top_products:
            day_qty = day_df2(final_df,name)
            data.append(go.Bar(name = name, x=day_qty.index, y= day_qty['quantity']))
        
        fig = go.Figure(data)
        fig.update_layout(barmode = 'stack')
        st.plotly_chart(fig, use_container_width=True)
        st.text('''Slight spike on Tuesdays across all products. Sales increase from 
Friday to Sunday generally, indicating the weekend might be a good time to conduct sales.''')

    day_bar_overall(top_products)

    st.subheader('Effect of Hour of Day on Quantity Sold')
    def df_hour_quantity2(data, name):
        hr_qty = data[data['name'] == name].groupby('hour').size()
        hr_qty_df = hr_qty.to_frame().reset_index()
        hr_qty_df.columns = ['hour', 'quantity']
        return hr_qty_df

    

    def bar_chart_hour_overall(top_products):
        data = []
        for name in top_products:
            hr_qty_df = df_hour_quantity2(final_df, name)
            data.append(go.Bar(name = name, x=hr_qty_df['hour'], y= hr_qty_df['quantity']))
        
        fig = go.Figure(data)
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
    bar_chart_hour_overall(top_products)

    st.subheader('Effect of Month on Quantity Sold')

    def df_month_quantity(data, name):
        month_qty = pd.DataFrame(data[data['name'] == name].groupby('month').size()).reset_index()
        month_qty.columns = ['month','quantity']
        return month_qty

    def bar_chart_month_overall(top_products):
        data = []
        for name in top_products:
            month_qty_df = df_month_quantity(final_df, name)
            data.append(go.Bar(name = name, x=month_qty_df['month'], y= month_qty_df['quantity']))
        
        fig = go.Figure(data)
        fig.update_layout(barmode = 'stack')
        st.plotly_chart(fig, use_container_width=True)
        st.text('''Based on this data from March till December, we can see that there are generally 
higher number of purchases in months July and August for each item. We can attribute this 
to the holiday season coming to an end, where both students & working adults get back to 
school / work and manufacturers are found to generally promote products during this 
back-to-school/work period''')

    bar_chart_month_overall(top_products)