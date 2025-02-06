#Cluster Model 2

#%%Stock Keyword Clusters Can Be Re-run as more stocks are added.
import pandas as pd
import re
import os
os.chdir('''Team_094\\SentimentModelScripts''')

#df = pd.read_csv('Outputs/WSB_Comments_Clean_wClusters.csv')
df = pd.read_csv('Outputs/WSB_Comments_Clean_wClusters_Reduced.csv')

df = df.fillna('')

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



#df.to_csv('Outputs/WSB_Comments_Clean_wClusters.csv',index=False)

df.to_csv('Outputs/WSB_Comments_Clean_wClusters_Reduced.csv',index=False)


