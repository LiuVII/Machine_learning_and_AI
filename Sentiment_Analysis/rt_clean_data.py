import numpy as np
import cPickle
from collections import defaultdict
import sys, re
import pandas as pd

import matplotlib.pyplot as plt

def build_data_cv(data_folder, cv=10, clean_string=True):
    """
    Loads data and split into 10 folds.
    """
    revs = []
    pos_file = data_folder[0]
    neg_file = data_folder[1]
    vocab = defaultdict(float)
    with open(pos_file, "rb") as f:
        for line in f:       
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"y":1, 
                      "text": orig_rev,                             
                      "num_words": len(orig_rev.split()),
                      "split": np.random.randint(0,cv)}
            revs.append(datum)
    with open(neg_file, "rb") as f:
        for line in f:       
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"y":0, 
                      "text": orig_rev,                             
                      "num_words": len(orig_rev.split()),
                      "split": np.random.randint(0,cv)}
            revs.append(datum)
    return revs, vocab
    
def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype='float32')            
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    return word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)  

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    # string = re.sub(r"[^A-Za-z0-9!?]", " ", string)     
    string = re.sub(r"[^A-Za-z!?]", " ", string) 
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip() if TREC else string.strip().lower()

def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)   
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()

if __name__=="__main__":    
    # w2v_file = sys.argv[1]     
    cv = 10
    data_folder = ["rt-polarity.pos","rt-polarity.neg"]    
    print "loading data...",        
    revs, vocab = build_data_cv(data_folder, cv=cv, clean_string=True)
    max_l = np.max(pd.DataFrame(revs)["num_words"])
    print "data loaded!"
    print "number of sentences: " + str(len(revs))
    print "vocab size: " + str(len(vocab))
    print "max sentence length: " + str(max_l)
    
    # Split data into train, validation and test sets
    test_id = np.random.randint(0,cv)
    validation_id = (test_id + np.random.randint(1,cv)) % cv
    
    clean_train = [[],[]]
    clean_validation = [[],[]]
    clean_test = [[],[]]

    for datum in revs:
        if datum['split'] == test_id:
            clean_test[0].append(datum['text'])
            clean_test[1].append(datum['y'])
        elif datum['split'] == validation_id:
            clean_validation[0].append(datum['text'])
            clean_validation[1].append(datum['y'])
        else:
            clean_train[0].append(datum['text'])
            clean_train[1].append(datum['y'])
    
    #Output clean data to file
    test_data = pd.DataFrame(data={'reviews': clean_test[0], 'sentiment': clean_test[1]})
    test_data.to_csv( "Clean_test.csv", index=False, quoting=3 )

    validation_data = pd.DataFrame(data={'reviews': clean_validation[0], 'sentiment': clean_validation[1]})
    validation_data.to_csv( "Clean_validation.csv", index=False, quoting=3 )

    train_data = pd.DataFrame(data={'reviews': clean_train[0], 'sentiment': clean_train[1]})
    train_data.to_csv( "Clean_train.csv", index=False, quoting=3 )

    print "loading word2vec vectors...",
    # w2v = load_bin_vec(w2v_file, vocab)
    # print "word2vec loaded!"
    # print "num words already in word2vec: " + str(len(w2v))
    # add_unknown_words(w2v, vocab)
    # W, word_idx_map = get_W(w2v)
    # rand_vecs = {}
    # add_unknown_words(rand_vecs, vocab)
    # W2, _ = get_W(rand_vecs)
    # cPickle.dump([revs, W, W2, word_idx_map, vocab], open("mr.p", "wb"))
    print "dataset created!"
    # Print bar chart for length of sentences
    # d = dict()
    # for el in pd.DataFrame(revs)["num_words"]:
    #     if el not in d:
    #         d[el] = 0
    #     else:
    #         d[el] += 1
    # x_pos = []
    # y_pos = []
    # data = [[value, key] for key, value in d.iteritems()]
    # print(data)
    # zipped = zip(*data)
    # x_pos = np.array(zipped[0])
    # y_pos = np.array(zipped[1])
    # print(x_pos.shape, y_pos.shape)
    # plt.bar(x_pos, y_pos)
    # plt.ylabel('Words')
    # plt.title('Number of words in reviews')
    plt.show()