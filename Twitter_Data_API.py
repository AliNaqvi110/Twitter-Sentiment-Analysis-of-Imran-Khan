import requests
import json
import numpy as np
import pandas as pd
import re
from pandas import json_normalize
from textblob import TextBlob
import spacy
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from spacytextblob.spacytextblob import SpacyTextBlob
import re
import nltk

# berear token
BEARER_TOKEN = "Your-API-Key"

def search_twitter(query, tweet_fields, bearer_token = BEARER_TOKEN):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}

    url = "https://api.twitter.com/2/tweets/search/recent?query={}&{}&max_results=100".format(
        query, tweet_fields
    )
    response = requests.request("GET", url, headers=headers)

    print(response.status_code)

    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()


query = "Imran Khan"
#twitter fields to be returned by api call
tweet_fields = "tweet.fields=text,author_id,created_at"

#twitter api call
json_response = search_twitter(query=query, tweet_fields=tweet_fields, bearer_token=BEARER_TOKEN)

only_data= json_response["data"]

df = json_normalize(only_data)
df['text'] = df['text'].apply(str)


nltk.download('words')
words = set(nltk.corpus.words.words())

##Cleaning the tweets
def cleaner(tweet):
    tweet = re.sub("@[A-Za-z0-9]+","",tweet) #Remove @ sign
    tweet = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", tweet) #Remove http links
    tweet = " ".join(tweet.split())
    
    tweet = tweet.replace("#", "").replace("_", " ") #Remove hashtag sign but keep the text
   
    return tweet
df['text']= df['text'].map(lambda x: cleaner(x))
df['created_at'] = pd.to_datetime(df['created_at'])
df = df.sort_values(by = 'created_at')

df['Year'] = df['created_at'].dt.year
df['Month'] = df['created_at'].dt.month
df['Date'] = df['created_at'].dt.date
df['time'] = df['created_at'].dt.time
df['hours'] = df['created_at'].dt.hour

nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('spacytextblob')

sentiment=[]



for i in df['text']:
    senti= nlp(i)._.blob.polarity      
    
    if senti >= 0.2:
        sentiment.append('Positive')
    elif senti <= -0.05:
        
        sentiment.append('Negative')
    else:
        sentiment.append('None')


          
df['sentiment']= sentiment
print(df.head())
df1= df.groupby("sentiment")["id"].count().reset_index()
print(df1.head())