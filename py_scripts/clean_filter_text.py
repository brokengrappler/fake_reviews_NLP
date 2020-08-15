import pandas as pd
import re
import string
import numpy as np

from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from nltk.tag import pos_tag
from nltk.corpus import stopwords

'''
Module:
1) Cleans text via regex
2) Tokenizes step 1
3) filters stop words and other custom words
4) lemmitize
5) Output pd.Series of rejoined text
'''

def cleaning_function(text):
    clean_text = re.sub(r'[^\x00-\x7f]',r' ',text)
    clean_text = re.sub(r'\w*((\w)\2{2,})\w*', ' ', clean_text)
    clean_text = re.sub('\w*\d\w*', ' ', clean_text)
    clean_text = re.sub('"', ' ',clean_text)
    clean_text = re.sub('[%s]' % re.escape(string.punctuation), ' ', clean_text)
    clean_text = clean_text.lower()
    return clean_text

def tokenizer(text):
    tbwt = TreebankWordTokenizer()
    text_out = tbwt.tokenize(text)
    return text_out

def filter_words(text_tok):
    '''
    Filter stop words and custom (hotel) words and all verbs
    '''
    filtered_word_list=[]
    pos_filter = ['VB', 'VBN','VBD', 'VBG', 'VBP', 'VBZ', 'NNP', 'NNPS']
    cust_sw = stopwords.words('english')
    hotel_sw = ['room','chicago','hotel','downtown', 'michigan']
    cust_sw.append(hotel_sw)
    for word in text_tok:
        if (word not in cust_sw) or (pos_tag([word])[0][1] not in pos_filter):
            filtered_word_list.append(word)
    return filtered_word_list

def stemmer(text_eng):
    ps = PorterStemmer()
    stem_text = [ps.stem(word) for word in text_eng if len(word)>2]
    return stem_text

def clean_text(text):
    '''
    Input pd.Series of text
    '''
    sample1 = text.apply(lambda x: cleaning_function(x))
    sample1_tok = sample1.apply(tokenizer)
    sample1_filtered = sample1_tok.apply(filter_words)
    sample1_stem = sample1_filtered.apply(stemmer)
    clean_out = sample1_stem.apply(lambda x: ' '.join(x))
    return clean_out

def main_clean():
    reviews = pd.read_csv('../raw_data/deceptive-opinion.csv')
    y = reviews['deceptive']
    x = reviews.drop('deceptive', axis=1)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=444)
    hotel_rev_cl = clean_text(x_tr['text'])
    return hotel_rev_cl

if __name__ == '__main__':
    main_clean(hotel_reviews)