import pandas as pd
from nltk.sentiment import vader
from textblob import TextBlob

def sentiment_translator(compound):
    '''
    Converts sentiment number to text evaluation (pos/neut/neg)
    :param compound:
        float score from vader
    :return:
        string pos/neut/neg
    '''
    if compound > .1:
        return 'pos'
    elif compound < -.1:
        return 'neg'
    else:
        return 'neut'

def vader_sentiment(text):
    '''
    Assesses sentiment of text
    :param text:
        Series of text
    :return:
        Data frame with sentiment scores and translation to negative, positive, neutral
    '''
    hotel_df = pd.DataFrame(text.values, columns=['review'])
    sid = vader.SentimentIntensityAnalyzer()
    hotel_df['sent_score'] = hotel_df['review'].apply(lambda x: sid.polarity_scores(x))
    hotel_df['compound'] = hotel_df['sent_score'].apply(lambda x: x['compound'])
    hotel_df['overall_sent'] = hotel_df['compound'].apply(lambda x: sentiment_translator(x))
    return hotel_df

def tb_sentiment(text):
    '''
    Assesses sentiment of text
    :param text:
        Series of text
    :return:
        Data frame with sentiment scores and translation to negative, positive, neutral
    '''
    text_df = pd.DataFrame(text.values, columns=['review'])
    text_df['polarity'] = text_df['review'].apply(lambda x: TextBlob(x).sentiment.polarity)
    text_df['subjectivity'] = text_df['subjectivity'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    return text_df

if __name__ == '__main__':

    file_df = pd.read_pickle('../pickles/hotel_reviews.pkl')
    print(vader_sentiment(file_df))