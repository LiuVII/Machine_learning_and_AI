from __future__ import print_function

import pandas as pd
import numpy as np
from collections import defaultdict

import matplotlib.pyplot as plt

# train_data = pd.read_csv("Clean_train_ns_ad.csv", names=['id', 'reviews'], quoting=3)
# validation_data = pd.read_csv("Clean_validation_ns_ad.csv", names=['id', 'reviews'], quoting=3)
# test_data = pd.read_csv("Clean_test_ns_ad.csv", names=['id', 'reviews'], quoting=3)
# combined_data = pd.read_csv("All_reviews_ns_ad.csv", names=['reviews'], quoting=3)
combined_data = pd.read_csv("rt-polarity.pos", names=['reviews'], quoting=3)
print(combined_data)
d = dict()
for el in combined_data:
	if len(el) not in d:
		d[len(el)] = 0
	else:
		d[len(el)] += 1
	# print(len(el), el)
	
x_pos = []
y_pos = []
data = [[value, key] for key, value in d.iteritems()]
print(data)
zipped = zip(*data)
x_pos = np.array(zipped[0])
y_pos = np.array(zipped[1])
print(x_pos.shape, y_pos.shape)
# plt.bar(data[:][0], data[:][1])
plt.bar(x_pos, y_pos)
plt.ylabel('Words')
plt.title('Number of words in reviews')
# plt.show()