import pandas as pd
from nltk.sentiment import vader

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

def calc_sentiment(text):
    hotel_df = pd.DataFrame(text.values, columns=['review'])
    sid = vader.SentimentIntensityAnalyzer()
    hotel_df['sent_score'] = hotel_df['review'].apply(lambda x: sid.polarity_scores(x))
    hotel_df['compound'] = hotel_df['sent_score'].apply(lambda x: x['compound'])
    hotel_df['overall_sent'] = hotel_df['compound'].apply(lambda x: sentiment_translator(x))
    return hotel_df

if __name__ == '__main__':

    file_df = pd.read_pickle('../pickles/hotel_reviews.pkl')
    print(calc_sentiment(file_df))