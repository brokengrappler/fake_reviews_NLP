import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from pprint import pprint
from sklearn.cluster import KMeans

import clean_filter_text as cft

def display_topics(model, feature_names, no_top_words, topic_names=None):
    topic_dict = {}
    for ix, topic in enumerate(model.components_):
        # if not topic_names or not topic_names[ix]:
        #     print("\nTopic ", ix)
        # else:
        #     print("\nTopic: '",topic_names[ix],"'")
        topic_list = [feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]
        # print(", ".join(topic_list))
        topic_dict[ix] = topic_list
    topic_count = [num for num in range(no_top_words)]
    topics_df = pd.DataFrame.from_dict(topic_dict, orient='index',
                                       columns=topic_count)
    return topics_df

def nmf_model(text_file, n=10, ngrams=(1,1)):
    tfidf = TfidfVectorizer(min_df=.05, max_df=.6, ngram_range=ngrams, stop_words='english')
    tdm = tfidf.fit_transform(text_file)
    nmf = NMF(n, max_iter=600)
    nmf_topic = nmf.fit_transform(tdm)
    return display_topics(nmf, tfidf.get_feature_names(), 5)
    #pprint(nmf_topic)

if __name__ == '__main__':
    file_df = cft.main_clean()
    ngrams = (1,2)
    print(nmf_model(file_df, 20, ngrams))
