#%% Clean Text
import pandas as pd
import numpy as np
import re
import nltk
import os
from tqdm import tqdm

os.chdir('''Team_094\\SentimentModelScripts''')

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

df = df.drop(df[df['ProcessedComments'] == 'deleted'].index)
#%% Import Cardiff University NLP model for Twitter Sentiment as a baseline.
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

# test = "I hate hedge funds"

# encoded_input = tokenizer(test, return_tensors='pt')
# output = model(**encoded_input)
# scores = output[0][0].detach().numpy()
# scores = softmax(scores)
# ranking = np.argsort(scores)
# ranking = ranking[::-1]
# labels = ['negative','neutral','positive']

# for i in range(scores.shape[0]):
#     l = labels[ranking[i]]
#     s = scores[ranking[i]]
#     print(f"{i+1}) {l} {np.round(float(s), 4)}")

#labels[ranking[0]]
#scores[ranking[0]]

def cardiff_tweet_sentiment(comments):
    predlabel = []
    predprob = []
    labels = ['negative','neutral','positive']
 #   for comment in range(0, len(comments)):

    processed = comments #(str(comments[comment]))
    encoded_input = tokenizer(processed, return_tensors='pt')
    try:
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        prediction = labels[ranking[0]]
        prob = scores[ranking[0]]
        predlabel.append(prediction)
        predprob.append(prob)
    except:
        prediction = ''
        prob = ''
        predlabel.append(prediction)
        predprob.append(prob)


    return predlabel,predprob



features = df.iloc[:,5].values

#features = features[:2]
#cardiff_tweet_sentiment(features[2])[0][0]
#cardiff_tweet_sentiment(features[2])[1][0]

label = [cardiff_tweet_sentiment(features[i])[0][0] for i in tqdm(range(len(features)))]

prob = [cardiff_tweet_sentiment(features[i])[1][0] for i in tqdm(range(len(features)))]

#cardiff_tweet_sentiment(features[3355])[0][0]

df['SentimentLabel'] = label
df['SentimentProb'] = prob



#%% Add Empty Label Column
df['HumanSentiment'] = ''

df2 = df.dropna()
#%% Output to csv
df2.to_csv('Outputs/WSB_Comments_Clean.csv',index=False)
