#%% Import Data

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import bertopic
from bertopic import BERTopic
import tensorflow as tf
import transformers
from transformers import TFAutoModel,RobertaTokenizer,TFRobertaModel
from keras.models import load_model

import pandas as pd
import numpy as np
import re
import nltk
from tqdm import tqdm
import logging

from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()




os.chdir('''Team_094\\SentimentModelScripts''')


#%% Read from ingested data and clean comments for inference
#df=pd.read_csv('Data/WSB_Comments_Delta.csv')
df=pd.read_csv('Data/WSB_Comments.csv')

def text_clean(comments):
    features = []
    for comment in range(0, len(comments)):
        # Replace ' with ''
        processed = (str(comments[comment])).replace("’","")
        processed = processed.replace("'","")

        # Remove all the special characters
        processed = re.sub(r'\W', ' ', processed)

        # Remove all single characters
        #processed= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed)

        # Remove single characters from the start
        #processed = re.sub(r'\^[a-zA-Z]\s+', ' ', processed)

        # Substituting multiple spaces with single space
        processed = re.sub(r'\s+', ' ', processed, flags=re.I)

        # Trim text
        processed = processed.strip()

        # Converting to Lowercase
        #processed = processed.lower()
        features.append(processed)
    return features

features = df.iloc[:,1].values
processed_features = text_clean(features)
df['ProcessedComments'] = processed_features

df['Date']=pd.to_datetime(df['Date Posted'], errors='coerce')
df = df.fillna('')
df = df[df['Date'].notna()]
df = df.drop(df[df['ProcessedComments'] == 'deleted'].index)
df = df.reset_index(drop = True)

#%% Predict Sentiment through Model API


bert = TFRobertaModel.from_pretrained("roberta-base")
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

def prep_data(text):
    tokens = tokenizer.encode_plus(text, max_length=512, truncation=True,
                                   padding='max_length', add_special_tokens=True,#return_token_type_id=False,
                                   return_tensors='tf')
    return{
        'input_ids': tf.cast(tokens['input_ids'], tf.float64),
        'attention_mask': tf.cast(tokens['attention_mask'], tf.float64)
        }

SentimentModel = load_model('Models/Model/sentiment_model_teamsentiment')

cor = df['ProcessedComments'].apply(prep_data)


pred = [np.argmax((SentimentModel.predict(cor[i]))[0]) for i in tqdm(range(len(cor)))]

df['Predicted_Sentiment'] = pred

df['Predicted_SentimentLabel'] = pd.cut(df['Predicted_Sentiment'], [-1,0,1,2], labels=['positive','negative','neutral'])
df['Predicted_SentimentValue'] = pd.cut(df['Predicted_Sentiment'], [-1,0,1,2], labels=['0.9','0.2','0.5'])
df['Predicted_SentimentValue'] = df['Predicted_SentimentValue'].astype(float)
del df['Predicted_Sentiment']

from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
    
df['Predicted_SentimentWeighted'] = df['Predicted_SentimentValue']*df['Score']
df['Predicted_SentimentWeighted'] = min_max_scaler.fit_transform(df[['Predicted_SentimentWeighted']])

#%% Predict Topic Cluster through Model API

from bertopic import BERTopic

#ClusterModel = BERTopic.load("Models/Model/InitialClusteringModel")
ClusterModel = BERTopic.load("Models/Model/ClusteringModel")
Comments = df['ProcessedComments'].to_list()

t,p = ClusterModel.transform(Comments)

df['Predicted_Cluster'] = t

#%% Cluster Topic Based on Reference to the Stock
df_reference = pd.read_csv('Dependencies/Stocks.csv')

stocksearch = pd.Series(df_reference['Stock_Search_List'].squeeze()).to_list()
stocksearchpattern = []

for item in stocksearch:
    itemlower = '\\b'+item+'\\b'
    itemupper = '\\b'+item.upper()+'\\b'
    stocksearchpattern.append(itemlower)
    stocksearchpattern.append(itemupper)
    
pattern = '|'.join(stocksearchpattern)

df['StockSearch_Cluster'] = df['ProcessedComments'].str.extractall(f"({pattern})").groupby(level=0).agg(','.join)
df['StockSearch_Cluster'] = df['StockSearch_Cluster'].fillna("NA")


#%%OUTPUT TO CSV

df.to_csv('Outputs/WSB_Comments_Delta_Inference_Reduced.csv',index=False)

#%% TO ADD 
#1. JOIN TO STOCK DATA 

df = pd.read_csv('Outputs/WSB_Comments_Delta_Inference_Reduced.csv')
df['Date']=pd.to_datetime(df['Date'], errors='coerce')
df = df.fillna('NA')


def stock_data(stockname, commentdata, stockreference, cluster_m):
    import yfinance as yf
    import datetime
    
    #Read stockname
    stockname_u = stockname.upper()
    stockname_l = stockname.lower()

    stockreference1 = stockreference[stockreference.Stock_Search_List == stockname_l]['Group'].to_list()[0]
    stockreference = stockreference[stockreference.Group == stockreference1]['Stock_Search_List'].to_list()
    
    stockname1 = stockreference[0]
    stockname2 = stockreference[1]
    
    #Pull Stock From Yahoo Finance
    date_1 = datetime.datetime.strptime('2021-10-23', '%Y-%m-%d').date()#datetime.date.today()
    start_date = datetime.datetime.strptime('2021-01-01', '%Y-%m-%d').date()#date_1 - datetime.timedelta(days=30)
    df_stock = yf.download(stockname_u, pd.to_datetime(start_date), date_1)
    df_stock.reset_index(level=0, inplace=True)
    df_stock['Stock'] = stockname_u
    
    #Pull Top Most Correlated Topics From Stock Name (if probability is greater than 0.3)
    topcluster1 = [cluster_m.find_topics(stockname1,top_n=3)[0][i] for i in range(len(cluster_m.find_topics(stockname1,top_n=3)[0])) if cluster_m.find_topics(stockname1,top_n=3)[1][i]>0.3]
    topcluster2 = [cluster_m.find_topics(stockname2,top_n=3)[0][i] for i in range(len(cluster_m.find_topics(stockname2,top_n=3)[0])) if cluster_m.find_topics(stockname2,top_n=3)[1][i]>0.3]
    #topcluster = set(topcluster1,topcluster2)
    topcluster = list(set(topcluster1 + topcluster2))

    # Filter Data For Correlated Clusters
    df_filter = df[df['Predicted_Cluster'].isin(topcluster)]
    df_filter['Stock'] = stockname_u
    
    #Get Weighted Average Sentiment From Correlated Clusters  i.e. Sentiment per comment, multiplied by the size of its audience
    df_filter['Predicted_SentimentWeighted'] = df_filter['Predicted_SentimentValue']*df_filter['Score']
    #Scale it between 0 and 1
    df_filter['Predicted_SentimentWeighted'] = min_max_scaler.fit_transform(df_filter[['Predicted_SentimentWeighted']])
    
    df_topic_g = df_filter.groupby(['Date'])['Predicted_SentimentWeighted'].mean().to_frame().reset_index()
    df_topic_g['Stock'] = stockname_u
    df_stock = df_stock.merge(df_topic_g,on = ['Stock','Date'], how='left')
    df_stock.Predicted_SentimentWeighted = df_stock.Predicted_SentimentWeighted.fillna(df_stock.Predicted_SentimentWeighted.mean())
    df_stock.Predicted_SentimentWeighted = min_max_scaler.fit_transform(df_stock[['Predicted_SentimentWeighted']])
   
    #Filter Data For Reference Clusters
    
    stockname1_u = stockname1.upper()
    stockname2_u = stockname2.upper()
    
    df_filter2 = df[(df.StockSearch_Cluster.str.contains(stockname1_u)) \
                   | (df.StockSearch_Cluster.str.contains(stockname2_u)) \
                   | (df.StockSearch_Cluster.str.contains(stockname1)) \
                   | (df.StockSearch_Cluster.str.contains(stockname2))] 
    df_filter2['Stock'] = stockname_u
    
    df_filter2['Predicted_SentimentWeighted'] = df_filter2['Predicted_SentimentValue']*df_filter2['Score']
    df_filter2['Predicted_SentimentWeighted'] = min_max_scaler.fit_transform(df_filter2[['Predicted_SentimentWeighted']])
    df_topic_g2 = df_filter2.groupby(['Stock', 'Date'])['Predicted_SentimentWeighted'].mean().to_frame().reset_index()
    
    df_stock = df_stock.merge(df_topic_g2,on = ['Stock','Date'], how='left')
    df_stock.Predicted_SentimentWeighted_y = df_stock.Predicted_SentimentWeighted_y.fillna(df_stock.Predicted_SentimentWeighted_y.mean())
    df_stock.Predicted_SentimentWeighted_y = min_max_scaler.fit_transform(df_stock[['Predicted_SentimentWeighted_y']])
    
    df_stock['StockSentiment'] = np.where(df_stock['Predicted_SentimentWeighted_x'].isna() & df_stock['Predicted_SentimentWeighted_y'].isna(), np.NaN, 
                       (np.where(df_stock['Predicted_SentimentWeighted_x'].isna(),df_stock['Predicted_SentimentWeighted_y'],
                                 (np.where(df_stock['Predicted_SentimentWeighted_y'].isna(),df_stock['Predicted_SentimentWeighted_x']
                                           ,df_stock['Predicted_SentimentWeighted_x']*0.3 + df_stock['Predicted_SentimentWeighted_y']*0.7))
                                 )))
    
    df_stock = df_stock.rename(columns={'Predicted_SentimentWeighted_x': 'Predicted_Cluster_Sentiment', 'Predicted_SentimentWeighted_y': 'Predicted_Reference_Sentiment'})

    return df_stock,df_filter,df_filter2
#%% Run scripts against all stocks

#df_completestock
#stockname = input("What stock are you interested in: ")
stockname = 'pltr'
try:
    df_stock = stock_data(stockname, commentdata = df, stockreference = df_reference, cluster_m = ClusterModel)[0]
    df_cluster = stock_data(stockname, commentdata = df, stockreference = df_reference, cluster_m = ClusterModel)[1]
    df_direct_reference_t = stock_data(stockname, commentdata = df, stockreference = df_reference, cluster_m = ClusterModel)[2]
except Exception as e:
    print(e)
    pass

df_referencetickers = pd.read_csv('Dependencies/StockTickers.csv')

df_completestock=pd.DataFrame()
df_cluster_reference=pd.DataFrame()
df_direct_reference=pd.DataFrame()

for i in df_referencetickers['Stock_Search_List']:
    print(i)
    try:
        df_completestock_prev = stock_data(i, commentdata = df, stockreference = df_reference, cluster_m = ClusterModel)[0]
        df_completestock = df_completestock.append(df_completestock_prev,ignore_index=True)
        
        df_cluster_reference_prev = stock_data(i, commentdata = df, stockreference = df_reference, cluster_m = ClusterModel)[1]
        df_cluster_reference = df_cluster_reference.append(df_cluster_reference_prev,ignore_index=True)

        df_direct_reference_prev = stock_data(i, commentdata = df, stockreference = df_reference, cluster_m = ClusterModel)[2]
        df_direct_reference = df_direct_reference.append(df_direct_reference_prev,ignore_index=True)
    except:
        pass




#%%OUTPUT TO CSV
#df_stock.to_csv('Outputs/ChosenStock_Prices_wSentiment_Reduced.csv',index=False)

df_completestock.to_csv('Outputs/Visualizations/Data/StocksPrices_wSentiment.csv',index=False)

df_cluster_reference.to_csv('Outputs/Visualizations/Data/Cluster_StocksComments.csv',index=False)

df_direct_reference.to_csv('Outputs/Visualizations/Data/Reference_StocksComments.csv',index=False)








