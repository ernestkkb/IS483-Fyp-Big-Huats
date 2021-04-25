import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
import re
import warnings
warnings.filterwarnings("ignore")
import math
import pandas as pd
import numpy as np
import scipy
import scipy.sparse
from tqdm import tqdm,tqdm_notebook
import pickle
import os
# Vis Libs..
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import joblib
# Text Libs..
import re
from gensim import corpora, models
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer,PorterStemmer
from nltk.stem.porter import *
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# ML Libs...
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from flask import Flask, render_template, request, redirect, url_for, jsonify
from joblib import load

# DL Libs..

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer() 
stop_words = stopwords.words('english')
def vectorize_data(vectorizer, data_column):
    return vectorizer.transform(data_column)

def fill_missing_data(data):
    data.brand_name.fillna(value = "Missing", inplace = True)
    data.category_name.fillna(value = "Missing", inplace = True)

def remove_invalid_prices(dataframe):
    #only prices above 3 are valid
    dataframe = dataframe[(dataframe.price >= 3)]

def add_log_price_column(dataframe):
    dataframe['log_price'] = np.log1p(dataframe['price'])

def tokenizer(text):
    if text:
        result = re.findall('[a-z]{2,}', text.lower())
    else:
        result = []
    return result

def preprocess(sentence):
    sentence=str(sentence)
    sentence=sentence.replace('{html}',"") 
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url=re.sub(r'http\S+', '',cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)  
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stop_words]
    stem_words=[stemmer.stem(w) for w in filtered_words]
    lemma_words=[lemmatizer.lemmatize(w) for w in stem_words]
    return " ".join(lemma_words)

def process_item_description(dataframe):
    dataframe['item_description'] = dataframe['item_description'].replace([np.nan],"No Description")
    dataframe['item_desc_processed']=dataframe['item_description'].map(lambda s:preprocess(s))

def generate_sentiment_scores(sentences):
    analyzer = SentimentIntensityAnalyzer()
    results = []
    for sentence in tqdm_notebook(sentences): 
        sentence_sentiment_score = analyzer.polarity_scores(sentence)
        results.append(sentence_sentiment_score)
    return results

def get_sentiments(dataframe):
    array = generate_sentiment_scores(dataframe['item_description'])
    negative, neutral, compound, positive = [], [], [], []
    for sentiment_dict in array:
        for sentiment, score in sentiment_dict.items():
            if(sentiment=='neg'):
                negative.append(score)
            elif(sentiment=='neu'):
                neutral.append(score)
            elif(sentiment=='compound'):
                compound.append(score)
            elif(sentiment=='pos'):
                positive.append(score)
    dataframe['negative'] = negative
    dataframe['positive'] = positive
    dataframe['neutral'] = neutral
    dataframe['compound'] = compound

def get_description_features(dataframe):
    percentage_of_exclamation_marks, percentage_of_star, percentage_of_ampersand, percentage_of_hashtag = [], [], [], []
    for description in dataframe['item_description']:
        hashtag_regex = re.compile(r'(#[a-z]{2,})')
        percentage_of_exclamation_marks.append(description.count('!')/len(description) * 100)
        percentage_of_star.append(description.count("*")/len(description)*100)
        percentage_of_ampersand.append(description.count("%")/len(description)*100)
        percentage_of_hashtag.append(len(hashtag_regex.findall(description))/len(description)*100)
    dataframe['percentage_of_exclamation_marks'] = percentage_of_exclamation_marks
    dataframe['percentage_of_star'] = percentage_of_star
    dataframe['percentage_of_ampersand'] = percentage_of_ampersand
    dataframe['percentage_of_hashtag'] = percentage_of_hashtag

def obtain_item_description_features(dataframe):
    get_sentiments(dataframe)
    get_description_features(dataframe)

def item_condition_shipping_dummy(dataframe):
    return scipy.sparse.csr_matrix(pd.get_dummies(dataframe[['item_condition_id', 'shipping']],
                                      sparse=True).values)
def replace_brand_name(dataframe):
    dataframe['brand_name'] = dataframe['brand_name'].replace([np.nan],"No Brand Name")

def create_pipe_line(new_input_dataframe, CV_category, name_vectorizer, tfidf_vectorizer, brand_name_vectorizer):

    fill_missing_data(new_input_dataframe)
    df_category_transformed = vectorize_data(CV_category, new_input_dataframe["category_name"])
    new_input_dataframe['name_process'] = new_input_dataframe['name'].map(lambda s:preprocess(s))
    name_transformed = vectorize_data(name_vectorizer, new_input_dataframe["name_process"])
    process_item_description(new_input_dataframe)
    item_desc_transformed = vectorize_data(tfidf_vectorizer, new_input_dataframe["item_desc_processed"])
    obtain_item_description_features(new_input_dataframe)
    item_shipping_dummies = item_condition_shipping_dummy(new_input_dataframe)
    replace_brand_name(new_input_dataframe)
    X_brand_transformed = vectorize_data(brand_name_vectorizer,new_input_dataframe['brand_name'])

    to_test = scipy.sparse.hstack((X_brand_transformed,
                               df_category_transformed,
                               item_desc_transformed,
                               item_shipping_dummies,
                               name_transformed,
                               np.array(new_input_dataframe['negative'])[:,None],
                               np.array(new_input_dataframe['neutral'])[:,None],
                               np.array(new_input_dataframe['compound'])[:,None],
                               np.array(new_input_dataframe['positive'])[:,None]
                              )).tocsr().astype('float32')
    return to_test

with open('required_files/CV_category.pkl', 'rb') as f:
    CV_category = pickle.load(f)
    
with open('required_files/name_vectorizer.pkl', 'rb') as f:
    name_vectorizer = pickle.load(f)

with open("required_files/tfidf_vectorizer.pkl", "rb") as f: 
    tfidf_vectorizer = pickle.load(f)

with open('required_files/brand_name_vectorizer.pkl', 'rb') as f:
    brand_name_vectorizer = pickle.load(f)

filename = 'required_files/finalized_model.sav'
tuned_model = joblib.load(filename)

def get_prediction(input, CV_category, name_vectorizer, tfidf_vectorizer, brand_name_vectorizer, tuned_model):
    new_input_dataframe = pd.DataFrame(input, columns=['name','item_condition_id','category_name', 'brand_name','shipping','item_description'])
    input_processed = create_pipe_line(new_input_dataframe, CV_category, name_vectorizer, tfidf_vectorizer, brand_name_vectorizer)
    price_list = np.expm1(tuned_model.predict(input_processed))
    return price_list[0].item()
# get_prediction([["black crop top coach", 1, "women", 'Prada', 1, "Brand new with tag bnwt free shipping "]], CV_category, name_vectorizer, tfidf_vectorizer, brand_name_vectorizer,tuned_model)


app = Flask(__name__)

# render default webpage
@app.route('/')
def home():
    return render_template('home.html')

# get the data for the requested query
@app.route('/getPrice', methods=['POST'])
def success():
    data = request.form.to_dict(flat=False)
    new_list = []
    for key, value in data.items():
        if value[0].isdigit():
            new_list.append(int(value[0]))
        else:
            new_list.append(value[0])
    predicted_price = get_prediction([new_list], CV_category, name_vectorizer, tfidf_vectorizer, brand_name_vectorizer,tuned_model)
    return str(round(predicted_price,2))

if __name__ == '__main__':
    app.run(debug=True)