import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import re

unl_data = pd.read_csv("unlabeledTrainData.tsv", header=0,\
 delimiter="\t", quoting=3)
data = pd.read_csv("labeledTrainData.tsv", header=0,\
 delimiter="\t", quoting=3)
test_data = pd.read_csv("testData.tsv", header=0,\
 delimiter="\t", quoting=3)
all_i = np.arange(len(data))
train_i, validation_i = train_test_split(all_i, train_size=0.8,\
 random_state = 43 )
#print(all_i, train_i, validation_i)
train = data.ix[train_i]
validation = data.ix[validation_i]
stops = set(stopwords.words("english"))

num_tr_reviews = train["review"].size
num_ts_reviews = validation["review"].size

def review_to_words(raw_review, stops, remove_stops=False, alpha_only=False):
	# Get HTML tags out
	bs_review = BeautifulSoup(raw_review, "html.parser").get_text()
	# Replace all non-letters to sapces
	if alpha_only:
		mean_chars = re.sub("[^a-zA-Z]", " ", bs_review)
	else:
		mean_chars = re.sub("[^a-zA-Z0-9]", " ", bs_review)
	# Lowercase and split to words
	words = mean_chars.lower().split()
	# Exclude stopwords
	if remove_stops:
		words = [w for w in words if (w == "not" or not w in stops)]
	return " ".join(words)

# Make clean train and validation reviews
clean_unlbl_reviews = []
clean_train_reviews = []
clean_validation_reviews = []
clean_test_reviews = []
for el in unl_data["review"]:
	clean_unlbl_reviews.append(review_to_words(el, stops))
for el in test_data["review"]:
	clean_test_reviews.append(review_to_words(el, stops))
for i in train_i:
	clean_train_reviews.append(review_to_words(train["review"][i], stops))
for i in validation_i:
	clean_validation_reviews.append(review_to_words(validation["review"][i], stops))

combined_reviews = clean_train_reviews + clean_validation_reviews + clean_test_reviews + clean_unlbl_reviews
combined_data = pd.DataFrame(data={"review": combined_reviews})
train_data = pd.DataFrame(data={"id":train_i, "sentiment": train["sentiment"], "review":clean_train_reviews})
validation_data = pd.DataFrame(data={"id":validation_i, "sentiment": validation["sentiment"], "review":clean_validation_reviews})
test_data = pd.DataFrame(data={"id":test_data["id"], "review":clean_test_reviews})

# Save cleaned data
combined_data.to_csv( "All_reviews_ns_ad.csv", index=False, header=0, quoting=3 )
train_data.to_csv( "Clean_train_ns_ad.csv", index=False, header=0, quoting=3 )
validation_data.to_csv( "Clean_validation_ns_ad.csv", index=False, header=0, quoting=3 )
test_data.to_csv( "Clean_test_ns_ad.csv", index=False, header=0, quoting=3 )