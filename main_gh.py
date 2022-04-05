# import libraries
import streamlit as st
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import os
import sys
import time, datetime
import json

import tweepy

import regex as re
import nltk

nltk.download('punkt', quiet = True)
nltk.download("stopwords", quiet = True)
nltk.download('omw-1.4', quiet = True)
nltk.download('vader_lexicon', quiet = True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import itertools
from itertools import chain
from collections import Counter
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image

# change directory
# os.chdir(r'C:\Users\Fahim Usman\Documents\Edu\Data Science\Python\code\NLP Project - Twitter Analytics')


st.markdown('<style>' + open('icon.css').read() + '</style>', unsafe_allow_html=True)

st.markdown('# Twitter Analytics')
st.write('\t*A small web app by Faheem Rajwadkar*')


# Importing the keys
# key = open('key.txt', 'r')
# consumer_key = key.readline().strip()
# consumer_secret = key.readline().strip()
# access_token = key.readline().strip()
# access_token_secret = key.readline().strip()
# key.close()

# get key from streamlit secrets
consumer_key = st.secrets.twitter_api.consumer_key
consumer_secret = st.secrets.twitter_api.consumer_secret
access_token = st.secrets.twitter_api.access_token
access_token_secret = st.secrets.twitter_api.access_token_secret


# Establish the connection with API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
# given authorization to tweepy for accessing the twitter data

api = tweepy.API(auth) # geting API


# Define variables, search term and number of tweets
search_term = st.text_input("Enter Keyword/Tag to search about: ")
no_of_tweets = st.number_input("Enter how many tweets to search: ", min_value = 100, max_value = 1000)

# Get no_of_tweets and search_term together
tweets = tweepy.Cursor(
    api.search_tweets, q=search_term + ' -filter:retweets -filter:links', 
    tweet_mode = "extended", lang = 'en',
    result_type = "mixed"
).items(no_of_tweets)

flag = False
dt = time.strftime("%Y%m%d")

if st.button("Retrieve Tweets"):
    try:
        # store the tweets into a list as json
        my_list_of_dicts = []
        for each_json_tweet in tweets:
            my_list_of_dicts.append(each_json_tweet._json)
            
        # write the tweets to a text file
        with open('tweets_' + search_term + dt + '.txt', 'w') as file:
            file.write(json.dumps(my_list_of_dicts, indent=4))
        
        flag = True
        
    except tweepy.errors.TooManyRequests:
        print("Oh No!!! We've reached the twitter request limit. Please wait for 15 minutes :'(")
        ph = st.empty()
        N = 15*60
        for secs in range(N,0,-1):
            mm, ss = secs//60, secs%60
            ph.metric("Countdown", f"{mm:02d}:{ss:02d}")
            time.sleep(1)

# Rest of the code
if flag == True:
    # create dataframe
    my_demo_list = []
    with open('tweets_' + search_term + dt + '.txt', encoding='utf-8') as json_file:  
        all_data = json.load(json_file)
        for each_dictionary in all_data:
            tweet_id = each_dictionary['id']
            text = each_dictionary['full_text']
            favorite_count = each_dictionary['favorite_count']
            retweet_count = each_dictionary['retweet_count']
            created_at = each_dictionary['created_at']
            
            user_name = each_dictionary['user']['screen_name']
            
            my_demo_list.append({'tweet_id': str(tweet_id),
                                
                                'user_name': str(user_name),
                                
                                'text': str(text),
                                'favorite_count': int(favorite_count),
                                'retweet_count': int(retweet_count),
                                'created_at': created_at,
                                })
            #print(my_demo_list)
            tweet_json = pd.DataFrame(my_demo_list, columns = 
                                    ['tweet_id', 'user_name', 'text', 
                                    'favorite_count', 'retweet_count', 
                                    'created_at'])
    
    tweets = tweet_json.loc[:, ['text', 'favorite_count', 'retweet_count']]

    
    # Remove emojis
    def remove_emoji(string):
        emoji_pattern = re.compile("["
                                u"\U0001F600-\U0001F64F"  # emoticons
                                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                u"\U00002500-\U00002BEF"  # chinese char
                                u"\U00002702-\U000027B0"
                                u"\U00002702-\U000027B0"
                                u"\U000024C2-\U0001F251"
                                u"\U0001f926-\U0001f937"
                                u"\U00010000-\U0010ffff"
                                u"\u2640-\u2642"
                                u"\u2600-\u2B55"
                                u"\u200d"
                                u"\u23cf"
                                u"\u23e9"
                                u"\u231a"
                                u"\ufe0f"  # dingbats
                                u"\u3030"
                                "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', string)


    # Remove Usernames
    def remove_username(string):
        return re.sub('@\w+\s?', '', string)

    hashtags = []
    for tweet in tweets:
        tweet_hash = []
        tweet_hash.append([re.sub('\s', '', x) for x in re.findall('#\w+\s?', tweet)])
        for hash in tweet_hash:
            if len(hash) > 0:
                hashtags.append(hash)
    
    # flatten above nested list
    def flatten(listOfLists):
        "Flatten one level of nesting"
        return chain.from_iterable(listOfLists)
    
    hashtags = list(flatten(hashtags))
    
    # get unique hashtags
    hashtags_unique = []
    for hash in hashtags:
        if hash not in hashtags_unique:
            hashtags_unique.append(hash)
        else:
            continue
        
    hash_counts = {}
    for hash in hashtags:
        if hash in hash_counts:
            hash_counts[hash] += 1
        else:
            hash_counts[hash] = 1

    # sort dict
    hash_counts_sorted = dict(sorted(hash_counts.items(), key=lambda item: item[1], reverse=True))
    
    remove_hashes = []

    for k, v in hash_counts_sorted.items():
        if v < 0.05*(len(tweets)):
            remove_hashes.append(k)
        else:
            continue
    
    def treat_hashtags(string):
        tokens = string.split()
        new_tokens = []
        for token in tokens:
            if token not in remove_hashes:
                if token in hashtags_unique:
                    new_tokens.append(re.sub(r'([a-z])([A-Z0-9])', r'\1 \2', token.replace('#', '')))
                else:
                    new_tokens.append(token)
            else:
                continue
        return ' '.join(new_tokens)
    
    # Combine above steps    
    df = tweets.loc[:, :]
    df['tweets'] = df['text'].apply(remove_emoji)
    df['tweets'] = df['tweets'].apply(remove_username)
    df['tweets'] = df['tweets'].apply(treat_hashtags)
    
    # save cleaned file
    # df.to_csv('cleaned_tweets_'+ search_term + dt +'.csv', index = False)  
    
    lemmatizer = WordNetLemmatizer()
    
    def clean_text(tweet):
        # Remove punctutation 
        tweet = re.sub(r'[^a-zA-Z0-9\s]','',tweet)
        
        # Convert to lower case
        tweet = tweet.lower()
        
        # tokenize
        tweet = tweet.split()
        
        # join the tokens
        return " ".join(tweet)
    
    df['cleaned_tweets'] = df['tweets'].apply(clean_text)
    
    # Sentiment Analysis using VADER
    sid = SentimentIntensityAnalyzer()
    
    def VADER_scores(tweet):
        return sid.polarity_scores(tweet)
    
    def VADER_sentiment(score):
        return score['compound']
    
    df['scroes_raw'] = df['text'].apply(VADER_scores)
    df['scores_tweets'] = df['tweets'].apply(VADER_scores)
    df['scores_cleaned_tweets'] = df['cleaned_tweets'].apply(VADER_scores)
    df['sentiment_tweets'] = df['scores_tweets'].apply(VADER_sentiment)
    df['sentiment_cleaned_tweets'] = df['scores_cleaned_tweets'].apply(VADER_sentiment)
    df['sentiment_raw'] = df['scroes_raw'].apply(VADER_sentiment)
    
    def get_sentiment(score):
        if score > 0.2:
            return "Positive"
        elif score < -0.2:
            return "Negative"
        else:
            return "Neutral"
    
    df['Sentiment'] = df['sentiment_cleaned_tweets'].apply(get_sentiment)
    
    df['true_score'] = df['sentiment_cleaned_tweets'] * (df['favorite_count']+1)
    
    # output to analyse
    # df.to_csv('sentiment_'+search_term+'.csv', index = False)

    # overall sentiment    
    overall_sentiment = df.groupby(['Sentiment']).sum()[['true_score', 'retweet_count', 'favorite_count']].reset_index()
    tweet_counts = df.groupby(['Sentiment']).count()['tweets'].reset_index()
    overall_sentiment = overall_sentiment.merge(tweet_counts, on = 'Sentiment')
    overall_sentiment['Sentiment Share (%)'] = overall_sentiment['favorite_count'] / overall_sentiment['favorite_count'].sum()
    overall_sentiment['Tweet Share (%)'] = overall_sentiment['tweets'] / overall_sentiment['tweets'].sum()

    text = (" ".join(text for text in df.cleaned_tweets))
    text = text.split()
    text = [word for word in text if word not in stopwords.words('english')]
    text = [word for word in text if len(word) > 2]
    
    word_freq = Counter(text)
    
    twitter_mask = np.array(Image.open("twitter_mask.png"))

    wordcloud = WordCloud(
        stopwords=STOPWORDS,
        background_color=None,
        max_words=200,
        color_func=lambda *args, **kwargs: (255,255,255),
        # width=1800,
        # height=1400,
        mask=twitter_mask
    ).generate_from_frequencies(word_freq)

    fig, ax = plt.subplots(figsize = (18,12))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    plt.savefig("wordcloud.png", transparent = True, bbox_inches='tight',pad_inches = 0)

    # pie chart
    colors = ['#FF3B3B', '#FFE354', '#3EDD1E']
    fig2, ax = plt.subplots(figsize= (5, 5))
    ax.pie(
        overall_sentiment['Sentiment Share (%)'], labels = overall_sentiment.Sentiment, 
        # wedgeprops = { 'linewidth' : 2.5, 'edgecolor' : 'white' },
        explode = [0.02, 0.02, 0.02],
        colors=colors,
        autopct='%1.1f%%',
        textprops=dict(color="w", style='italic', weight = 'bold')
    )
    ax.texts[1].set_color('black')
    ax.texts[3].set_color('black')
    ax.texts[5].set_color('black')
    fig2.suptitle('Sentiment Share', color="w", style='italic', weight = 'bold')
    plt.savefig("fig2.png", transparent=True, bbox_inches='tight',pad_inches = 0)

    # pie chart 2
    colors = ['#FF3B3B', '#FFE354', '#3EDD1E']
    fig3, ax = plt.subplots(figsize= (5, 5))
    ax.pie(
        overall_sentiment['Tweet Share (%)'], labels = overall_sentiment.Sentiment, 
        # wedgeprops = { 'linewidth' : 2.5, 'edgecolor' : 'white' },
        explode = [0.02, 0.02, 0.02],
        colors=colors,
        autopct='%1.1f%%',
        textprops=dict(color="w", style='italic', weight = 'bold')
    )
    ax.texts[1].set_color('black')
    ax.texts[3].set_color('black')
    ax.texts[5].set_color('black')
    fig3.suptitle('Tweet Share', color="w", style='italic', weight = 'bold')
    plt.savefig("fig3.png", transparent=True, bbox_inches='tight',pad_inches = 0)

    st.markdown('### Word Cloud')
    wordcloud = Image.open('wordcloud.png')
    st.image(wordcloud, width = None)

    st.markdown('### Sentiment')

    col1, col2 = st.columns(2)
    with col1:
        sentiment = Image.open('fig2.png')
        st.image(sentiment, width = None)

    with col2:
        tweet = Image.open('fig3.png')
        st.image(tweet, width = None)

else :
    pass

