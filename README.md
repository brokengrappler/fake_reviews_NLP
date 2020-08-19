# Fake Yelp Review Identifier

### Description 

Use NLP, supervised, and unsupervised machine learning methods to identify fake yelp reviews.
58,716 N Flagged
8,303 Y Flagged
Data set provided appears to be flagging not only potential fraudulent reviews but also reviews that violate the terms of Yelp. 

I've used the deceptive review data set on hotels as found on Kaggle to create a model that:
1) Classifies the hotel reviews by topic using NMF matrix decomposition on a document term matrix created using TFIDF
2) Use KMeans clustering to cluster the reviews by the topics
3) Overlay the clustering vs. genuine/fake review labeling of the hotel data set
4) Based on features correlated with the clustering found in step 3, create a classifier matrix
5) Pipeline yelp reviews into the model to determine probability of fraudulent review

### Data Sources

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

Length of avg tweet as min: https://www.prnewsonline.com/whats-the-ideal-length-of-a-tweet/



### File Contents


### Dependencies



### Contributors:




