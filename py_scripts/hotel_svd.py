import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation as ldamodel
from gensim import corpora, models, similarities, matutils
from pprint import pprint
from nltk.corpus import stopwords
from sklearn.cluster import KMeans

import hotel_sentiment

def display_topics(model, feature_names, no_top_words, topic_names=None):
    for ix, topic in enumerate(model.components_):
        if not topic_names or not topic_names[ix]:
            print("\nTopic ", ix)
        else:
            print("\nTopic: '",topic_names[ix],"'")
        print(", ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

def lsa_model(vect, tdm, n=10):
    lsa = TruncatedSVD(n)
    lsa_topic = lsa.fit_transform(tdm)
    print(f'LSA explained VR:{lsa.explained_variance_ratio_}')
    display_topics(lsa, vect.get_feature_names(), 5)

def nmf_model(vect, tdm, n=20):
    nmf = NMF(n, max_iter=600)
    nmf_topic = nmf.fit_transform(tdm)
    display_topics(nmf, vect.get_feature_names(), 5)
    #pprint(nmf_topic)

def sklda_model(vect, tdm, n=10):
    lda = ldamodel(n_components=n, max_iter=100)
    lda_topic = lda.fit_transform(tdm)
    display_topics(lda, vect.get_feature_names(), 5)

def gslda_model(vect, tdm, n=10):
    corpus = matutils.Sparse2Corpus(tdm.T)
    id2word = dict((v, k) for k, v in vect.vocabulary_.items())
    lda = models.LdaModel(corpus=corpus, num_topics=n, id2word=id2word, passes=100)
    pprint(lda.print_topics())

def custom_stops():
    eng_sw = set(stopwords.words('english'))
    hotel_sw = ['room','chicago','hotel','downtown']
    #return list(eng_sw.update(hotel_sw))
    return eng_sw.update(hotel_sw)

if __name__ == '__main__':

    file_df = pd.read_pickle('../pickles/hotel_reviews.pkl')
    tfidf = TfidfVectorizer(min_df=.05, max_df=.6, ngram_range=(1,1), stop_words='english')
    tdm = tfidf.fit_transform(file_df)
    #lsa_model(tfidf, tdm, 20)
    nmf_model(tfidf, tdm, 20)
    #sklda_model(tfidf, tdm, 20)
    #gslda_model(tfidf, tdm, 15)
