import praw
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import re
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

nltk.download('stopwords')
nltk.download('punkt')

# a. Select the number of target classes (2 or 3 if we include neutral).
num_classes = 3
# b. Clean data and remove stopwords
stop_words = set(stopwords.words('english'))


def main():
    # scrape_reddit('war', 20000)

    df = pd.read_csv("./data/war.csv")

    # b. Clean data and remove stopwords
    df = df.drop(columns=['title'])
    df.drop_duplicates(subset=['comment'], inplace=True)

    df['comment'] = df['comment'].apply(lambda x: clean_text(x))
    df['comment'].replace('', np.nan, inplace=True)
    df = df.dropna(how='all')

    # c. Create word embeddings for vectorized representation of words
    # simillar in meaning // OR we use pretrained model
    # for language of choice
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['comment'])

    # d. Use K-MEANS to create clusters and use k=2 or k=3
    # depending on the numer of target classes
    kmeans = KMeans(n_clusters=num_classes, random_state=0)
    kmeans.fit(X)

    # e. Based on clusters tag data
    df['label'] = kmeans.labels_

    print(df.head())
    df.to_csv('./data/result.csv', index=False)


# Scrapping reddit in r/war
# Data acquisition concerns the collection of tweets.
# Each person scraps tweets (about 20k) to create a dataset for further processing.
# Tweets should be about current events, such as the war, NATO etc.
def scrape_reddit(subreddit, limit):
    reddit = praw.Reddit(
        client_id='',
        client_secret='',
        username='',
        password='',
        user_agent=''
    )
    posts = []
    i = 0
    j = 0
    for post in reddit.subreddit(subreddit).hot(limit=limit):
        i += 1
        submission = reddit.submission(post)
        print(f"{submission.comments.list()}")
        for comments in submission.comments.list():
            try:
                for comment in comments.body.split(". "):
                    j += 1
                    print(f"{i} {j}")
                    posts.append([post.title, comment])
            except:
                print("no object")
    df = pd.DataFrame(posts, columns=['title', 'comment'])
    df.to_csv(f'{subreddit}.csv', index=False)


# b. Clean data and remove stopwords
def clean_text(text):
    text = str(text)
    text = re.sub(r'\W+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if not token in stop_words]
    text = ' '.join(tokens)
    return text


main()
