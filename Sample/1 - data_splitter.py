# Take Data After Cleaning and Clustering - Split it into 7 parts for manual labelling.
import pandas 

df2 = pd.read_csv('Data/WSB_Comments_Clean_wClusters_Reduced.csv')


df2['Date']=pd.to_datetime(df2['Date Posted'])

months = [g for n, g in df2.groupby(pd.Grouper(key='Date',freq='M'))]


Alexandre = months[5].sample(250).append(months[6].sample(250))
Ayman = months[7].sample(250).append(months[8].sample(250))
Jordan = months[0].sample(250).append(months[1].sample(250))
Jonathan = months[2].sample(250).append(months[3].sample(250))
Reena = months[3].sample(250).append(months[4].sample(250))
Sid = months[9].sample(500)

Alexandre = Alexandre.to_csv('Sample/Raw/Alexandre.csv')
Ayman = Ayman.to_csv('Sample/Raw/Ayman.csv')
Jordan = Jordan.to_csv('Sample/Raw/Jordan.csv')
Jonathan = Jonathan.to_csv('Sample/Raw/Jonathan.csv')
Reena = Reena.to_csv('Sample/Raw/Reena.csv')
Sid = Sid.to_csv('Sample/Raw/Sid.csv')
