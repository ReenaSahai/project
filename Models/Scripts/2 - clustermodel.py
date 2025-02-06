from bertopic import BERTopic
import pandas as pd
import os
from tqdm import tqdm

os.chdir('''Team_094\\SentimentModelScripts''')

model = BERTopic(nr_topics='auto')

df = pd.read_csv('Outputs/WSB_Comments_Clean.csv')
df = df.fillna('')

docs = df['ProcessedComments']

topics,probs = model.fit_transform(docs)

model.save("Models/InitialClusteringModel")

#model1 = BERTopic.load("Models/InitialClusteringModel")

topics_df = pd.DataFrame({'Topic':topics}).reset_index()
topic_names = model.get_topic_info()

topics_df = topics_df.merge(topic_names, on = 'Topic', how = 'left')
topics_df = df.reset_index().merge(topics_df,on='index',how= 'left')
topics_df = topics_df.drop(columns=['index'])
topics_df = topics_df.drop(columns=['Count'])

df['Date']=pd.to_datetime(df['Date Posted'])
time = df['Date'].to_frame()
topics_df = topics_df.join(time)

RepText = model.get_representative_docs('')
RepText[-1] = ''
RepText = pd.DataFrame(RepText.items(),columns=['Topic','RepText'])

topic_names = topic_names.merge(RepText, on='Topic', how='left')


#%%Output to csv
topics_df.to_csv('Outputs/WSB_Comments_Clean_wClusters.csv',index=False)
topic_names.to_csv('Outputs/ExploratoryData/ClusteredTopicsSummary.csv',index=False)


#%%Check Most Associated Clusters
similar_topics,similarity = model.find_topics('gamestop',top_n=5)

model.find_topics('gme',top_n=5)

#%%Visualize
t = model.visualize_topics()
t.write_html('Outputs/Visualizations/clusters.html')

t=model.visualize_hierarchy()
t.write_html('Outputs/Visualizations/clusterhierachy.html')

times = df['Date'].to_list()
topics_over_time = model.topics_over_time(docs,topics,times)

topics_over_time.to_csv('Outputs/ExploratoryData/ClusteredTopicsOverTime.csv',index=False)


#%% Decrease number of clusters

ClusterModel = BERTopic.load("Models/Model/InitialClusteringModel")


#topics,probs = ClusterModel.fit_transform(docs)
#ClusterModel.get_topic_info()

# Reduce Topic to 1/25 size
new_topics, new_probs = ClusterModel.reduce_topics(docs, topics, probabilities=probs, nr_topics=25)
#del topics
#del probs

ClusterModel.save("Models/Model/ClusteringModel")

ClusterModel.find_topics('palantir',top_n=5)

topics_df = pd.DataFrame({'Topic':new_topics}).reset_index()
topic_names = ClusterModel.get_topic_info()

topics_df = topics_df.merge(topic_names, on = 'Topic', how = 'left')
topics_df = df.reset_index().merge(topics_df,on='index',how= 'left')
topics_df = topics_df.drop(columns=['index'])
topics_df = topics_df.drop(columns=['Count'])

df['Date']=pd.to_datetime(df['Date Posted'])
time = df['Date'].to_frame()
topics_df = topics_df.join(time)

RepText = ClusterModel.get_representative_docs('')
RepText[-1] = ''
RepText = pd.DataFrame(RepText.items(),columns=['Topic','RepText'])

topic_names = topic_names.merge(RepText, on='Topic', how='left')


#%%Output to csv
topics_df.to_csv('Outputs/WSB_Comments_Clean_wClusters_Reduced.csv',index=False)
topic_names.to_csv('Outputs/ExploratoryData/ClusteredTopicsSummary_Reduced.csv',index=False)

topic_names.to_csv('Outputs/Visualizations/Data/ClusteredTopicsSummary.csv',index=False)

#%% Visualize
t = ClusterModel.visualize_topics()
t.write_html('Outputs/Visualizations/clusters_r.html')

t=ClusterModel.visualize_hierarchy()
t.write_html('Outputs/Visualizations/clusterhierachy_r.html')

times = df['Date'].to_list()
topics_over_time = ClusterModel.topics_over_time(docs,new_topics,times)

topics_over_time.to_csv('Outputs/ExploratoryData/ClusteredTopicsOverTime_Reduced.csv',index=False)

topics_over_time.to_csv('Outputs/Visualizations/Data/TopicsOverTime.csv',index=False)


