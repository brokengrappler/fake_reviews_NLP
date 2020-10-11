
from textblob import TextBlob

def tb_subjectivity(text):
    sub_score = TextBlob(text).sentiment.subjectivity
    return sub_score

def tb_sentiment(text):
    pol_score = TextBlob(text).sentiment.polarity
    return 0 if pol_score < .3 else 1

def add_features_to_topics(topic_df, x_sel, keep_text=False):
    '''
    Takes topic model matrix and adds subjectivity, review length, and dummifies polarity. Also
    adds original index as a column in the topic model matrix.
    '''
    topic_df.index = x_sel.index
    x_mod = topic_df.join(x_sel['text'], how='left')
    x_mod['tb_subjective'] = x_mod['text'].apply(tb_subjectivity)
    #x_mod['review_length'] = x_mod['text'].apply(lambda x: len([x for x in x.split()]))
    try:
        x_mod['polarity'] = x_sel['polarity'].map({'negative':0, 'positive': 1})
    except:
        x_mod['polarity'] = x_mod['text'].apply(tb_sentiment)
    return x_mod.drop('text', axis=1)

