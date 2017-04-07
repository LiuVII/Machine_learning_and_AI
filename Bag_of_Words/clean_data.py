import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import re

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

def review_to_words(raw_review, stops):
	# Get HTML tags out
	bs_review = BeautifulSoup(raw_review, "html.parser").get_text()
	# Replace all non-letters to sapces
	letters_only = re.sub("[^a-zA-Z]", " ", bs_review)
	# Lowercase and split to words
	words = letters_only.lower().split()
	# Exclude stopwords
	words = [w for w in words if (w == "not" or not w in stops)]
	return " ".join(words)

# Make clean train and validation reviews
clean_train_reviews = []
clean_validation_reviews = []
clean_test_reviews = []
for el in test_data["review"]:
	clean_test_reviews.append(review_to_words(el, stops))
for i in train_i:
	clean_train_reviews.append(review_to_words(train["review"][i], stops))
for i in validation_i:
	clean_validation_reviews.append(review_to_words(validation["review"][i], stops))

train_data = pd.DataFrame(data={"id":train_i, "sentiment": train["sentiment"], "review":clean_train_reviews})
validation_data = pd.DataFrame(data={"id":validation_i, "sentiment": validation["sentiment"], "review":clean_validation_reviews})
test_data = pd.DataFrame(data={"id":test_data["id"], "review":clean_test_reviews})

# Save cleaned data
train_data.to_csv( "Clean_train.csv", index=False, header=0, quoting=3 )
validation_data.to_csv( "Clean_validation.csv", index=False, header=0, quoting=3 )
test_data.to_csv( "Clean_test.csv", index=False, header=0, quoting=3 )