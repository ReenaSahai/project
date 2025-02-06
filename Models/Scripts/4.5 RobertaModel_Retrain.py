#%% Import Data

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import bertopic
from bertopic import BERTopic
import tensorflow as tf
import transformers
from transformers import TFAutoModel,RobertaTokenizer,TFRobertaModel
from keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
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
df=pd.read_csv('Data/WSB_Comments_Delta.csv')

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
#%% Retrain Model if needed

#PARTITION AND SPLIT DATA
df2 = df[['ProcessedComments', 'HumanSentiment']]

test = df['HumanSentiment'].value_counts()

df2['sentiment_numeric']  = pd.factorize(df2["HumanSentiment"])[0]

dictionary = pd.Series(df2["HumanSentiment"].values,index=df2['sentiment_numeric']).to_dict()
print(dictionary)
np.save('sentiments.npy', dictionary)

df2 = df2[['ProcessedComments', 'sentiment_numeric']]
print(df2.head())
print(len(df2))

from sklearn.model_selection import train_test_split
df_subset_train_model, df_subset_val_model = train_test_split(df2, test_size=0.1)


df2 = df_subset_train_model[['ProcessedComments','sentiment_numeric']]
print(len(df2))

#%ROBERTA SETUP
seq_len = 512
num_samples = len(df2)

Xids = np.zeros((num_samples, 512))
Xmask = np.zeros((num_samples, 512))

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

#from transformers import AutoTokenizer, TFAutoModel
#tokenizer = AutoTokenizer.from_pretrained("kamalkraj/deberta-base")

for i, phrase in enumerate(df2['ProcessedComments']):
    tokens = tokenizer.encode_plus(phrase, max_length=seq_len, truncation=True,
                                   padding='max_length', add_special_tokens=True,
                                   return_tensors='tf')
    Xids[i, :] = tf.cast(tokens['input_ids'],tf.float64)
    Xmask[i, :] = tf.cast(tokens['attention_mask'],tf.float64)

arr = df2['sentiment_numeric'].values
print(arr)

labels = np.zeros((num_samples, arr.max()+1))
print(labels.shape)

labels[np.arange(num_samples), arr] = 1

#PREPARE DATA FOR TF MODEL
import tensorflow as tf

dataset = tf.data.Dataset.from_tensor_slices((Xids,Xmask,labels))

def map_func(input_ids, masks, labels):
    return {'input_ids': input_ids, 'attention_mask': masks}, labels

dataset = dataset.map(map_func)

batch_size = 13
dataset = dataset.shuffle(10000).batch(batch_size,drop_remainder=True)

split = 0.80
size = int((num_samples/batch_size) * split)

train_ds = dataset.take(size)
val_ds = dataset.skip(size)
print(train_ds)
print(val_ds)
del dataset

#%%
history = SentimentModel.fit(
    train_ds,
    validation_data=val_ds,
    epochs = 100,
    callbacks=[early_stopping]
)

#SAVE MODEL
SentimentModel.save('Models/Model/sentiment_model_teamsentiment')
