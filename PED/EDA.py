import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.pyplot import figure
from IPython.display import display, HTML
import streamlit as st
from scipy import stats

st.title('Exploratory Data Analysis')
st.set_option('deprecation.showPyplotGlobalUse', False)

@st.cache(ttl = 3600)
def read_file (filename):
    data = pd.read_csv(filename, encoding = 'ISO-8859-1')
    return data

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
def summary(pid):
    df = data[data['StockCode'] == pid]
    df = df.reindex(df.index.repeat(df.Quantity))
    grouping = df.groupby(['UnitPrice']).size()
    grouping_df = grouping.to_frame().reset_index()
    grouping_df.columns = ['UnitPrice','Quantity']
    return pd.DataFrame(grouping_df)

for pid in chosen:
    st.text(data[data['StockCode'] == pid].Description.iloc[0])
    st.dataframe(summary(pid))

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
    data = data[(np.abs(stats.zscore(data['Quantity'])) < 3)]
    

#Function Definitions
def df_datepricequantity(data, stockID):
    date_price_qty = data[data['StockCode'] == stockID].groupby([data[data['StockCode'] == stockID].index, 'UnitPrice']).size()
    date_price_qty_df = date_price_qty.to_frame().reset_index()
    date_price_qty_df.columns = ['dates', 'price', 'quantity']
    return date_price_qty_df

def price_distribution_chart(data, stockID):
    data[data['StockCode'] == stockID]['UnitPrice'].plot(linewidth=0.5, figsize=(15,10))
    st.line_chart(data[data['StockCode'] == stockID]['UnitPrice'])

def boxplot_month(data, stockID):
    plt.figure(figsize=(15, 10))
    fig = sns.boxplot(data=data[data['StockCode'] == stockID], x='Month', y='UnitPrice')
    st.pyplot()

def boxplot_day(data, stockID):
    plt.figure(figsize=(15, 10))
    fig = sns.boxplot(data=data[data['StockCode'] == stockID], x='Weekday Name', y='UnitPrice', order = ['Monday', 'Tuesday', 'Wednesday','Thursday','Friday','Saturday','Sunday']);
    st.pyplot()

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
    plt.figure(figsize=(15, 10))
    fig = plt.bar(hr_qty_df['hour'], hr_qty_df['quantity'], color ='maroon',  width = 0.4) 
    plt.xlabel("Hour") 
    plt.ylabel("Quantity") 
    plt.title("Number of items sold per hour") 
    plt.show()

    st.pyplot() 

def df_date_quantity(data, stockID):
    date_qty = pd.DataFrame(data[data['StockCode'] == stockID].groupby(data[data['StockCode'] == stockID].index).size().reset_index())
    date_qty.columns = ['dates','quantity']
    return date_qty

def bar_chart_date(date_qty):
    plt.figure(figsize=(15, 10))
    fig = plt.bar(date_qty['dates'], date_qty['quantity'], color ='maroon',  width = 1.5) 
    plt.xlabel("Date") 
    plt.ylabel("Quantity")
    plt.title("Date of highest purchase") 
    plt.show()
    st.pyplot()

def price_chart(data, stockID):
    percentiles = data[data['StockCode'] == stockID]['UnitPrice'].quantile([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, .9, 0.95, 1 ])
    plt.plot(percentiles)
    plt.figure(figsize=(15, 10))
    plt.title('Price Percentile Distribution')
    plt.xlabel("Percentile") 
    plt.ylabel("Price") 
    plt.show()
    st.line_chart(percentiles)

def quartile_barchart(data, stockID):
    data = data[data['StockCode'] == stockID]
    data['Quantile'] = pd.qcut(data['UnitPrice'], q=np.arange(0,1.1,0.1), duplicates='drop')
    df_tempo = pd.DataFrame(data.groupby('Quantile').agg('size').reset_index())
    df_tempo.columns = ['Quantile','Quantity']

    sns.set(rc={'figure.figsize':(15,10)}, style = 'whitegrid')
    fig = sns.barplot(x = "Quantile", y = "Quantity", data = df_tempo, palette = "hls")
    plt.title('Histogram by Price Percentiles')
    plt.show()
    st.pyplot()

def day_df(data,stockID):
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Sunday"]
    day_qty = pd.DataFrame(data[data['StockCode'] == stockID].groupby('Weekday Name').size().reindex(order))
    day_qty.columns = ['quantity']
    return day_qty

def day_bar(data, stockID):
    # order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Sunday"]
    
    # day_qty.columns = ['day','quantity']
    # day_qty = day_qty.set_index('day').loc[order]
    # st.dataframe(day_qty)
    df = day_df(data,stockID)
    fig = sns.barplot(x = df.index, y = "quantity", data = df, palette = "hls")
    st.pyplot()

def display_item(data, stockID):
    price_chart(data, stockID)
    quartile_barchart(data, stockID)
    display(HTML(day_df(data, stockID).head().to_html()))
    day_bar(data, stockID)
    display(HTML(df_datepricequantity(data,stockID).head().to_html()))
    price_distribution_chart(data, stockID)
    boxplot_month(data, stockID)
    boxplot_day(data, stockID)
    rolling_mean(data, stockID)
    display(HTML(df_hour_quantity(data, stockID).head().to_html()))
    bar_chart_hour(df_hour_quantity(data, stockID))
    display(HTML(df_date_quantity(data, stockID).head().to_html()))
    bar_chart_date(df_date_quantity(data, stockID))

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
    st.subheader('Trends in Pricing Strategy')
    rolling_mean(data, id)
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
st.text('It has been observed that there is increasing demand of items bought between 10am - 12pm and occasionally early afternoons at 2-3pm.')
st.subheader('Pricing Strategy')
st.text('The second half of the year (from September/October) is often a good period to steadily increase your prices while keeping the quantity of items bought high.')
st.subheader('Weekdays > Weekends for Household Products')
st.text('It is better to have any kind of sale from Monday - Thursday as it has shown that these days have better receptivity compared to Friday - Sunday (Weekend)')

st.header('Electronic products')
df = read_file('category_price.csv')
st.subheader('Preview of Data')
st.dataframe(data.head())


st.subheader('Data Preprocessing')

with st.echo():
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
ax1 = sns.boxplot( x = df_clean['disc_price'] , y = df_clean['Category_name'], orient = "h", order = ranks)
ax1.set(xlim = (0, 6000))
ax1.set(title = "Price Distribution per Category ")
plt.rcParams['figure.figsize'] = [60, 90]
plt.rcParams['font.size'] = 50
sns.set_style("darkgrid")
st.pyplot()

#Detect top selling items
top = df_clean.groupby('name').size().reset_index(name='counts').nlargest(15,['counts'])
top_products = top['name'].tolist()

df_clean = df_clean[df_clean['name'].isin(top_products)]
st.dataframe(df_clean)

st.subheader('Price Distribution Plot for top selling products')
plt.figure(figsize=(30,50))
plot_number = 1
for name, selection in df_clean.groupby('name'):
    ax = plt.subplot(15,3, plot_number)
    sns.distplot( df_clean.loc[df_clean.name == name, "disc_price"] , color="dodgerblue")
    ax.set(title = name[:30], xlabel = 'Price',ylabel = 'Frequency')

    # Go to the next plot for the next loop
    plot_number = plot_number + 1
plt.tight_layout()
st.pyplot()

st.subheader('Dropping Price Outliers')
from scipy import stats
with st.echo():
    frames = []
    for i in top_products:
        test = df_clean[df_clean.name.str.contains(i)]
        test = test[(np.abs(stats.zscore(test['disc_price'])) < 3)]
        frames.append(test)

    final_df = pd.concat(frames)
    st.dataframe(final_df)

# Price Distribution Plot for top selling products
plt.figure(figsize=(30,50))
plot_number = 1
for name, selection in final_df.groupby('name'):
    ax = plt.subplot(15,3, plot_number)
    sns.distplot( final_df.loc[final_df.name == name, "disc_price"] , color="dodgerblue")
    ax.set(title = name[:30], xlabel = 'Price',ylabel = 'Frequency')

    # Go to the next plot for the next loop
    plot_number = plot_number + 1
    

plt.tight_layout()
st.pyplot()