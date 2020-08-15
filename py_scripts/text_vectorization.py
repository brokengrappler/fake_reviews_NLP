import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import NMF
from sklearn.decomposition import TruncatedSVD
from gensim import corpora, models, similarities, matutils

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
    nmf = NMF(n)
    nmf_topic = nmf.fit_transform(tdm)
    display_topics(nmf, vect.get_feature_names(), 5)

def lda_model(tfidf, dtm):
    id2word = dict((v,k) for k, v in tfidf.vocabulary_.items())
    corpus = matutils.Sparse2Corpus(dtm.T)
    lda = models.LdaModel(corpus=corpus, num_topics=5, id2word=id2word, passes=10)
    print(lda.print_topics())

if __name__ == '__main__':

    #file_df = pd.read_pickle('../pickles/hotel_reviews.pkl')
    row_lim = 200000
    max_words = 50000
    tfidf = TfidfVectorizer(min_df=.05, max_df=.6, ngram_range=(1,2), stop_words='english')
    #line below is for when I had to limit due to crashing
    #dtm  = tfidf.fit_transform(file_df[:row_lim])
    tdm = tfidf.fit_transform(file_df)
    #lsa_model(tfidf, tdm, 20)
    nmf_model(tfidf, tdm, 10)
    #lda_model(tfidf, dtm)

