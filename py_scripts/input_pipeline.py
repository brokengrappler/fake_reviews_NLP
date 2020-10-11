import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

class nmf_obj:
    def __init__(self):
        self.tfidf = None
        self.tdm = None
        self.nmf = None
        self.nmf_topic = None
        self.topic_df = None
        self.topic_dict = {}
        self.feature_names = None

    def init_tfidf(self, min_df=.05, max_df=.6, ngram=(1, 1),
                                     stop_words='english'):
        self.tfidf = TfidfVectorizer(min_df=min_df, max_df=max_df, ngram_range=ngram,
                                     stop_words=stop_words)

    def init_tdm(self, text_file):
        self.tdm = self.tfidf.fit_transform(text_file)
        self.feature_names = self.tfidf.get_feature_names()

    def init_nmf(self, n=10):
        self.nmf = NMF(n, max_iter=1000, random_state=444)
        self.nmf_topic = self.nmf.fit_transform(self.tdm)

    def display_topics(self, no_top_words=5, topic_names=None):
        for ix, topic in enumerate(self.nmf.components_):
            topic_list = [self.feature_names[i]
                          for i in topic.argsort()[:-no_top_words - 1:-1]]
            self.topic_dict[ix] = topic_list
        topic_count = [num for num in range(no_top_words)]
        self.topics_df = pd.DataFrame.from_dict(self.topic_dict, orient='index',
                                                columns=topic_count)
        print(self.topics_df)

    def export_topic_df(self):
        topic_col_list=[]
        for v in self.topic_dict.values():
            topic_col_list.append(','.join(v))
        return pd.DataFrame(self.nmf_topic, columns=topic_col_list)