import pandas as pd
import numpy as np
import streamlit as st
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

from bokeh.palettes import Spectral3
from bokeh.models.tools import HoverTool
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource
################################################################################################################################ Function Definitions ####################################################################################
@st.cache(allow_output_mutation=True)
def read_file(filename):

    if(filename == "data.csv"):
        data = pd.read_csv(filename, encoding = 'ISO-8859-1')

    else:
        data = pd.read_csv(filename)

    return data

def summary(final_df, name):
    df = final_df[final_df['name'] == name]
    grouping = df.groupby(['disc_price']).size()
    grouping_df = grouping.to_frame().reset_index()
    grouping_df.columns = ['UnitPrice','Quantity']
    return pd.DataFrame(grouping_df)

def ols_summary_log_skewQ(data):
    model = ols("np.log(Quantity) ~ UnitPrice", data = data).fit()
    st.text(model.summary())

def ols_summary_log_skewP(data):
    model = ols("Quantity ~ np.log(UnitPrice)", data = data).fit()
    st.text(model.summary())

def ols_summary_log_skewBoth(data):
    model = ols("np.log(Quantity) ~ np.log(UnitPrice)", data = data).fit()
    st.text(model.summary())

def campaign_check(final_df):
    if final_df['disc_price'] < final_df['price']:
        val = 1
        #campaign
    else:
        val = 0
        # no campaign

    return val
def context_ols_c(final_df, name1, i):
    testing = final_df.loc[(final_df['name'] == name1) & (final_df['campaign'] == 1)]
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
    
def context_ols_nc(final_df, i):
    testing = final_df.loc[(final_df['name'] == i) & (final_df['campaign'] == 0)]
    tester = testing.groupby(['disc_price']).size()
    test_df = tester.to_frame().reset_index()
    test_df.columns = ['UnitPrice','Quantity']
    #print(test_df)
    
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

def summary_nocampaign(final_df, name):
    testing = final_df.loc[(final_df['name'] == name) & (final_df['campaign'] == 0)]
    tester = testing.groupby(['disc_price']).size()
    test_df = tester.to_frame().reset_index()
    test_df.columns = ['UnitPrice','Quantity']
    return pd.DataFrame(test_df)

def ped_context(final_df, products):
    newDict = {
        'Item': [],
        'PED (Overall)':[],
        'PED (No Campaign)': []
    }
    
    for name in products: 
        data = summary(final_df, name)
        model = ols("np.log(Quantity) ~ UnitPrice", data = data).fit()
        intercept, slope = model.params
        mean_price = np.mean(data['UnitPrice'])
        mean_quantity = np.mean(data['Quantity'])
        price_elasticity = ((slope) * (mean_price/mean_quantity))

        newDict['Item'].append(name)
        newDict['PED (Overall)'].append(price_elasticity)
        
        data = summary_nocampaign(final_df, name)
        model = ols("np.log(Quantity) ~ UnitPrice", data = data).fit()
        intercept, slope = model.params
        mean_price = np.mean(data['UnitPrice'])
        mean_quantity = np.mean(data['Quantity'])
        price_elasticity = ((slope) * (mean_price/mean_quantity))
        newDict['PED (No Campaign)'].append(price_elasticity)
        
    
    return pd.DataFrame(newDict)
################################################################################################################################ Function Definitions ####################################################################################
def app():
    st.header('Order Volume Impact on Campaigns')
    st.text('''
    Being an established ecommerce platform, Shopee often hold a multitude of campaigns to
    engage its consumers. Through this analysis, we hope to be able to find out if sales 
    campaigns can affect the price elasticity demand of a product.''')
    top_products = ['JBL Clip2 Portable Speaker',
                    'Yamaha - Natural Sound 5 2-Way All-Weather Outdoor Speakers (Pair) - White"', 
                    'Russound - Acclaim 5 Series 6-1/2 2-Way Indoor/Outdoor Speakers (Pair) - White"',
                    'MCR-B043 30W Bluetooth Wireless Music System (Black)',
                    'Kicker DSC44 4 D-Series 2-Way Car Speakers with 1/2" Tweeters"',
                    'Alpine - 6-1/2 2-Way Component Car Speakers with Poly-Mica Cones (Pair) - Black"',
                    'Details About Alpine 400w 5.25 Typee Coaxial 2way Car Speakers | Spe5000"']
    df = read_file('category_price.csv')
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
    df_clean = df_clean[df_clean['name'].isin(top_products)]


    frames = []
    for i in top_products:
        test = df_clean[df_clean['name'] == i]
        test = test[(np.abs(stats.zscore(test['disc_price'])) < 3)]
        frames.append(test)

    final_df = pd.concat(frames)
    final_df['campaign'] = final_df.apply(campaign_check, axis=1)
################################################################################################################################ Main Code ####################################################################################
    test = pd.DataFrame(final_df.groupby(['name','campaign']).size(), columns = ['count'])
    st.dataframe(test.reset_index())
    st.text('''
    Due to the lack of data depicting campaign transactions which makes it unrepresentative 
    for regression modelling, we will focus on comparing the overall data for each product 
    (i.e. during campaign + non campaign) vs during non campaign sales. The following three
    products with campaign periods identified are shown with its respective regression models.
    ''')
    products_check = ['Alpine - 6-1/2 2-Way Component Car Speakers with Poly-Mica Cones (Pair) - Black"',
                 'JBL Clip2 Portable Speaker',
                 'Kicker DSC44 4 D-Series 2-Way Car Speakers with 1/2" Tweeters"']

    for i in products_check:
        st.subheader(i)
        context_ols_nc(final_df, i)

    main = ped_context(final_df, products_check)

    difference_dict = {
        'Item' : [],
        'Difference %' : []
    }

    st.dataframe(main)
    for index, row in main.iterrows():
        difference_dict['Item'].append(row['Item'])
        overall = row['PED (Overall)']
        nocampaign = row['PED (No Campaign)']
        difference = nocampaign - overall 
        difference_dict['Difference %'].append((difference/abs(overall))*100)

    campaignped = pd.DataFrame(difference_dict)

    st.dataframe(campaignped)

    item = campaignped['Item']
    difference = campaignped['Difference %']

    source = ColumnDataSource(data=dict(item=item, difference=difference, color=Spectral3))

    max_y = max(difference)
    min_y = min(difference)

    p = figure(x_range=item, y_range=(min_y + min_y * 0.2 ,max_y + max_y *0.2), plot_height=1000, title="Percentage change in PED for Campaigns",
        toolbar_location=None, tools="")

    p.vbar(x='item', top='difference', width=0.9, color='color', source=source)

    p.xgrid.grid_line_color = None
    p.legend.orientation = "horizontal"
    p.legend.location = "top_center"
    p.xaxis.major_label_orientation = 'vertical'
    p.add_tools(HoverTool(tooltips=[("Percentage Difference", "@difference")]))
    st.bokeh_chart(p, use_container_width=True)

    st.text('''

    1) Alpine - 6-1/2 2-Way Component Car Speakers with Poly-Mica Cones (Pair) - Black": PED of 
    0.657(from 0.675) indicates that a 10% increase in the price for the light holder results
    in a 6.75% increase in quantity demanded. [Absolute value of PED is less than 1 so its 
    quantity demanded is price inelastic.] So, by only considering the non campaign period, 
    we see that quantity demanded is more price inelastic than overall.

    2) JBL Clip2 Portable speaker: PED of 0.354 (from 0.460) indicates that a 10% increase in 
    the price for the jar results in a 3.54% increase in quantity demanded. [Absolute value of 
    PED is less than 1 so its quantity demanded is price inelastic.] So, by only considering 
    the non campaign period, we see that quantity demanded is more price inelastic than overall.

    3) Kicker DSC44 4 D-Series 2-Way Car Speakers with 1/2" Tweeters": PED of -1.417 
    (from -1.619) indicates that a 10% increase in the price for the popcorn holder results in 
    a 14% decrease in quantity demanded. [Absolute value of PED is more than 1 so its quantity 
    demanded is price elastic.] So, by only considering the non campaign period, we see that 
    quantity demanded is less price elastic than overall.

    We can see that PED (No Campaign) values are slightly lower than PED (overall) values 
    across all 3 products which have campaign and non campaign periods. People are less 
    sensitive to price changes during non campaign periods than during campaign periods.
    ''')