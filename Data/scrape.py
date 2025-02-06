"""
Approach
1. Use PMAW to extract title and information of WallStreetBets Daily and Weekend Discussion Threads
2. Use PRAW to extract comments (with some obvious initial preprocessing steps) from those posts
3. Create a dataframe to store the data that we would need and export it to a CSV file
"""

from pmaw import PushshiftAPI # extracts information (but cannot extract comments) from Reddit posts
from datetime import datetime # converts UTC-formatted time (a piece of information extracted by PMAW) to month-day-year
import random # creates the small sample of WSB posts to extract comments for midterm
import praw # extracts comments from WSB posts; posts can be narrowed down using PMAW
import pandas as pd # creates and stores information in dataframe that will be exported to CSV


api = PushshiftAPI()

# creates a 'submission' iterable object to search for Daily Discussion Threads with the following query - can edit query
daily_submissions = api.search_submissions(after=int(datetime(2021, 1, 1).timestamp()), # (YYYY, MM, DD); returns data starting at this date
                                           subreddit='wallstreetbets',
                                           title='Daily Discussion Thread for 2021', # what's entered is loosely searched (think Google search)
                                           fields=['created_utc','id','num_comments','title','url'], # remove this line if you want to see all fields as script is running
                                           sort_type='created_utc',
                                           sort='desc')


# to store the Daily Discussion Threads (dictionaries) initially
posts_raw = []

# enter text in the [] to filter out unwanted posts with titles that include any of the text
for submission in daily_submissions:
  if any(substring in submission['title'] for substring in ['Unpinned','Part','Pt.','#2','version','30th']) == False:
    posts_raw.append(submission)
    print(submission)

# number of Daily Discussion Threads
daily_count = len(posts_raw)
print('Number of Daily Discussion Threads -', daily_count)

      
# creates a 'submission' iterable object to search for Weekend Discussion Threads with the following query - can edit query
weekend_submissions = api.search_submissions(after=int(datetime(2021, 1, 1).timestamp()), # (YYYY, MM, DD); returns data at this start date
                                             subreddit='wallstreetbets',
                                             title='Weekend Discussion Thread for 2021', # what's entered is loosely searched (think Google search)
                                             fields=['created_utc','id','num_comments','title','url'], # remove this line if want to see all fields after running next cell (for-loop)
                                             sort_type='created_utc',
                                             sort='desc')

# enter text in the [] to filter out unwanted posts with titles that include any of the text
for submission in weekend_submissions:
  if any(substring in submission['title'] for substring in ['discussion','-']) == False:
      posts_raw.append(submission)
      print(submission)

# number of Weekend Discussion Threads
len(posts_raw) - daily_count

      
# removes duplicate threads and keeps the one with the larger number of comments
# posts_1 and posts_2 are copies of posts_raw in case posts_raw is needed for reference
# posts_final is the list of all threads with their selected information to be used for extracting comments next
posts_1 = posts_2 = sorted(posts_raw, key=lambda sub: sub['created_utc'])
posts_final = []

for submission_1 in posts_1:
  for submission_2 in posts_2:
    if submission_1['title'] == submission_2['title'] and submission_1['id'] != submission_2['id']:
      if submission_1['num_comments'] > submission_2['num_comments']:
        posts_final.append(submission_1)
      else:
        posts_final.append(submission_2)
      posts_1.remove(submission_1)
      posts_2.remove(submission_2)    
  if submission_1 in posts_1 and submission_1 not in posts_final:
    posts_final.append(submission_1) 

print(posts_final)

# number of Daily and Weekend Discussion Threads without duplicates
len(posts_final)

      
# takes 5 samples of threads posted from 1/1/2021 to current time with at least 100 comments (some threads have single-digit number of comments)
# posts_to_sample = [submission for submission in posts_final if submission['num_comments'] >= 100]
# samples = random.sample(posts_to_sample, 5)
# samples

      
# establish connection to Reddit API using PRAW to create 'submission' iterable object for extracting comments in next cell
reddit = praw.Reddit(client_id='gnbaQGt7tQHNFF1QCie0dA',
                     client_secret='qeG8GsbZRuEm5PHl0fIl43ehK0grkA',
                     user_agent='WSB Scraper (by u/Aggressive-Risotto)')

# dataframe to store data and that will be exported to CSV 
df = pd.DataFrame(columns=['Submission Title','Comment','Date Posted','Author','Score'])

# data stored in this order - thread title, comment, date posted for comment, comment author, comment score
# filtered out - comments made by VisualMod (MOD), deleted comments, comments with 0 score
# score is the sum of upvotes (+) and downvotes (-)
for post in posts_final:
  submission = reddit.submission(id=post['id'])
  submission.comments.replace_more(limit=0)
  for top_level_comment in submission.comments:
    if top_level_comment.author != 'VisualMod' and top_level_comment != '[deleted]' and top_level_comment.score > 0:
      df = df.append({'Submission Title': submission.title,
                      'Comment': top_level_comment.body,
                      'Date Posted': datetime.fromtimestamp(top_level_comment.created_utc).strftime('%m-%d-%Y'),
                      'Author': top_level_comment.author,
                      'Score': top_level_comment.score},
                     ignore_index=True)

print(df)
    
# specifically to check total number of comments
print(df.shape)
      
df.to_csv('WSB_Comments.csv', index=False)
