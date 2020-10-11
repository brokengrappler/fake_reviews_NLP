# Fake Yelp Review Identifier

## Description 

Use NLP, supervised, and unsupervised machine learning methods to identify patterns in fraudulent Yelp reviews for restaurants.

The primary file is a jupyter notebook (see File Contents for further information) that:
1) performs topic modeling on a corpus of hotel reviews via NMF matrix docomposition on a doc-term matrix created using TFIDF
2) uses KMeans clustering to cluster topics
3) based on analyzing fake/genuine labels vs. clustering, create a classifier matrix
4) pipeline Yelp reviews into steps 1-3 above to compare reviews flagged in model vs. reviews flagged by Yelp's algorithm

#### Added Objective/Notebook

In an attempt to see if it was possible to make the clustering sector agnostic (work for both hotel/hospitality and restaurants), I used word embeddings on the reviews to see if there were other interesting clusterings and insights.

#### Additional Background

The project started as an attempt to identify fraudulent Yelp reviews based on the data set described below. Upon additional analysis of the original data set, it appeared that the labels for fake/genuine weren't always exclusively for fraudulent reviews, but also included reviews flagged for violation of Yelp's terms. Given I had a 2 week turnaround and I discovered this 5 days into the window, I did my best to adapt using the labeled data for fraudulent hotel reviews.

## File Contents

#### Primary Files
**Fake_Rev_Pipeline.ipynb:** Performs steps 1-4 described in the description section, including outputing results (confusion matrix and F1 score). Additional dependent Python scripts are described below.
**Fake_w2v_Imp.ipynb:** Notebook performing the added objective above of clustering and classifying on word embedding.

#### Supporting Files
**clean_filter_text.py**: Cleans, tokenizes, filters custom stop words, and lemmitize reviews
**topic_meta_data.py**: Adds subjectivity and sentiment features for each review via textblob
**input_pipeline.py**: Contains class with methods to perform topic modeling for a corpus
**w2v_clean_filter.py**: Sames as clean_filter_text.py catered to word2vec implementation

## Dependencies
gensim==3.8.3
pandas==1.0.5
textblob==0.15.3
nltk==3.4.4
numpy==1.18.5
scikit_learn==0.23.2

## Data Sources

#### Fake Hotel Review (https://www.kaggle.com/rtatman/deceptive-opinion-spam-corpus)

This corpus contains:

400 truthful positive reviews from TripAdvisor (described in [1])
400 deceptive positive reviews from Mechanical Turk (described in [1])
400 truthful negative reviews from Expedia, Hotels.com, Orbitz, Priceline, TripAdvisor and Yelp (described in [2])
400 deceptive negative reviews from Mechanical Turk (described in [2])
Each of the above datasets consist of 20 reviews for each of the 20 most popular Chicago hotels (see [1] for more details). The files are named according to the following conventions:
Directories prefixed with fold correspond to a single fold from the cross-validation experiments reported in [1] and [2].

[1] M. Ott, Y. Choi, C. Cardie, and J.T. Hancock. 2011. Finding Deceptive Opinion Spam by Any Stretch of the Imagination. In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies.

[2] M. Ott, C. Cardie, and J.T. Hancock. 2013. Negative Deceptive Opinion Spam. In Proceedings of the 2013 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies.

#### Yelp Data (http://liu.cs.uic.edu/download/yelp_filter/)

From: Arjun Mukherjee, Vivek Venkataraman, Bing Liu, and Natalie Glance. What Yelp Fake Review Filter Might Be Doing. Proceedings of The International AAAI Conference on Weblogs and Social Media (ICWSM-2013), July 8-10, 2013, Boston, USA.

Labels: 
Reviews with Y/N: Reviews obtained from the restaurant page wherein we get all Y reviews from the filtered section and N reviews from the regular page.

Reviews with YR/NR: Reviews obtained from the reviewer profile page. These reviews are not just for restaurants but for every business the reviewer put a review for. We used it to identify how many of his reviews were filtered. The YR is determined by whether the review was availble on that particular business page. If it wasnt present (we determine this by crawling every page for that business exhaustively) we gave it YR as in we assumed it was filtered. If it was present it was given a NR value.

In the paper, "What Yelp Fake Review Filter might be Doing?" (ICWSM'13), the author only used reviews with label Y and N.

