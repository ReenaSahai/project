#After Labelling Completed - Rejoin Data, Standardize it and output it as a csv to be used to build sentiment model

import pandas as pd
import os
import re
os.chdir('''Team_094\\SentimentModelScripts''')

df1 = pd.read_csv('Sample/Labelled/Alexandre.csv')
df2 = pd.read_csv('Sample/Labelled/Ayman.csv')
df3 = pd.read_csv('Sample/Labelled/Jonathan.csv', encoding='cp1252')
df4 = pd.read_csv('Sample/Labelled/Jordan.csv')
df5 = pd.read_csv('Sample/Labelled/Reena.csv')
df6 = pd.read_csv('Sample/Labelled/Sid.csv')

df = pd.concat([df1,df2,df3,df4,df5,df6])



df = df[['Comment','ProcessedComments', 'HumanSentiment']]




df['HumanSentiment']=df['HumanSentiment'].str.replace(r'\bg\b','positive')\
                                         .str.replace(r'\bn\b','negative')\
                                         .str.replace(r'\bne\b','negative')\
                                         .str.replace(r'\bneg\b','negative')\
                                         .str.replace(r'\bb\b','negative')\
                                         .str.replace(r'\bneu\b','neutral')\
                                        .str.replace(r'\bnetural\b','neutral')\
                                        .str.replace(r'\bneu\b','neutral')\
                                        .str.replace(r'\bpos\b','positive')\
                                        .str.replace(r'\boos\b','positive')\
                                        .str.replace(r'\bPositive\b','positive')
#df = pd.read_csv('Outputs/WSB_Comments_Clean_wClusters.csv')
df = df.fillna('positive')

df.HumanSentiment.value_counts()



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


df.to_csv('Sample/Labelled/data.csv',index=False)
