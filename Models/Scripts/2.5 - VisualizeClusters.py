from bertopic import BERTopic
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from umap import UMAP


os.chdir('''Team_094\\SentimentModelScripts''')


ClusterModel = BERTopic.load("Models/Model/ClusteringModel")

# BERTopic Code to pull cluster x and y coordinates - taken from BERtopic library

topics = None
top_n_topics: int = None


if topics is not None:
    topics = list(topics)
elif top_n_topics is not None:
    topics = sorted(ClusterModel.get_topic_freq().Topic.to_list()[1:top_n_topics + 1])
else:
    topics = sorted(list(ClusterModel.get_topics().keys()))
    
topic_list = sorted(topics)
frequencies = [ClusterModel.topic_sizes[topic] for topic in topic_list]
words = [" | ".join([word[0] for word in ClusterModel.get_topic(topic)[:5]]) for topic in topic_list]


all_topics = sorted(list(ClusterModel.get_topics().keys()))
indices = np.array([all_topics.index(topic) for topic in topics])
embeddings = ClusterModel.c_tf_idf.toarray()[indices]
embeddings = MinMaxScaler().fit_transform(embeddings)
embeddings = UMAP(n_neighbors=2, n_components=2, metric='hellinger').fit_transform(embeddings)

df = pd.DataFrame({"x": embeddings[1:, 0], "y": embeddings[1:, 1],
                   "Topic": topic_list[1:], "Words": words[1:], "Size": frequencies[1:]})


df.to_csv('Outputs/ExploratoryData/ClustersMapping_R.csv',index=False)
df.to_csv('Outputs/Visualizations/Data/ClustersMapping.csv',index=False)
