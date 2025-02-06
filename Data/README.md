Runtime - <50 minutes (~45 minutes during step to extract, process, and store each comment for each post in dataframe)

**WSB_Comments.csv**<br>
Outputted CSV file from scrape script<br>
Contains 5 columns in following order - Submission Title, Comment, Date Posted, Author, Score (sum of upvotes and downvotes) <br>
100704 comments

**scrape.ipynb**<br>
Contains code to scrape comments from Daily Discussion Threads and Weekend Discussion Threads from WallStreetBets<br>
_Contains outputs and more descriptive comments_ - meant for thorough rundown of code for reader<br>
Link to Google Colab notebook - https://colab.research.google.com/drive/1G2b9u0Tq2H0MtPTqDXANliBEbyERm8se?usp=sharing<br>

**scrape.py**<br>
Same essential code as in scrape.ipynb; some non-essential pieces of code were left off or adjusted<br>
_Contains less descriptive comments_

**Requirements**<br>
Python 3.7.12<br>
pandas==1.1.5<br>
pmaw==2.1.0<br>
praw==7.4.0
