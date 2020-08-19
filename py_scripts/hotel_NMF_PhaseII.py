import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from pprint import pprint

import clean_filter_text as cft

def display_topics(model, feature_names, no_top_words, topic_names=None):
    topic_dict = {}
    for ix, topic in enumerate(model.components_):
        topic_list = [feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]
        topic_dict[ix] = topic_list
    topic_count = [num for num in range(no_top_words)]
    topics_df = pd.DataFrame.from_dict(topic_dict, orient='index',
                                       columns=topic_count)
    return topics_df, topic_dict

def nmf_model(text_file, n=5, ngrams=(1,1), print_topics=True):
    '''
    returns n number of words
    :param text_file:
    :param n:
    :param ngrams:
    :param print_topics:
    :return:
    '''
    topic_col_list = []
    tfidf = TfidfVectorizer(min_df=.05, max_df=.7, ngram_range=ngrams,
                            stop_words='english')
    tdm = tfidf.fit_transform(text_file)
    nmf = NMF(n, max_iter=1000, random_state=444)
    nmf_topic = nmf.fit_transform(tdm)
    topic_df, topic_dict = display_topics(nmf, tfidf.get_feature_names(), 5)
    # trying to put topic list in 1 string to label NMF dataframe
    for v in topic_dict.values():
        topic_col_list.append(','.join(v))
    if print_topics:
        print(topic_df)
    return pd.DataFrame(nmf_topic, columns=topic_col_list)

if __name__ == '__main__':
    file_df = cft.main_clean()
    ngrams = (1,2)
    print(nmf_model(file_df, 15, ngrams))
