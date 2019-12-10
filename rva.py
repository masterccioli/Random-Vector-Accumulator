# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 19:18:10 2019

@author: masterccioli
"""

# time coding performance
from time import time
start = time()

# import statements
from scipy import sparse
import numpy as np
import pandas as pd

# load/prep training file
corpus_file_name = 'tasasentdocs.txt'
with open(corpus_file_name,'r') as f:
    corpus = f.read()
    
corpus = corpus.split('\n')
corpus = [i.split(' ') for i in corpus if len(i.split(' ')) > 1 ]

flattened = [c for corp in corpus for c in corp]

# initialize indexing dictionary
words = sorted(set(flattened))
words = dict(zip(words, range(0, len(words))))

# initialize sparse WxD matrix
col = list(map(words.get, flattened))
row = [i for index,corp in enumerate(corpus) for i in [index]*len(corp)]
data = [1] * len(row)
wd = sparse.csr_matrix((data, (row, col)), shape=(len(corpus),len(words)),dtype='float32').transpose()

# get WxD matrix
ww = wd.dot(wd.transpose()) # dot of WxD DxW yields WxW
ww.setdiag(0) # don't count words as co-occuring with self
ww.eliminate_zeros() # simplify data structure

# get cosine matrix for words in word list
Word_list = ["financial","savings","finance","pay","invested",
               "loaned","borrow","lend","invest","investments",
               "bank","spend","save","astronomy","physics",
               "chemistry","psychology","biology","scientific",
               "mathematics","technology","scientists","science",
               "research","sports","team","teams","football",
               "coach","sport","players","baseball","soccer",
               "tennis","basketball"]

compare = ww[[words[i] for i in Word_list]]

# apply random vectors
n = 1000 # number of dimensions
context_vectors = np.random.normal(0,1/np.sqrt(n),(len(words), n))

compare = compare.dot(context_vectors) # transform sparse WxW into RVA

# get cosine matrix comparing words in Word_list
out_cosine_matrix = compare.dot(compare.transpose()) / np.outer(np.sqrt(np.power(compare,2).sum(1)),np.sqrt(np.power(compare, 2).sum(1)))
out_cosine_matrix = pd.DataFrame(out_cosine_matrix)
out_cosine_matrix.to_csv('cosine_ww_matrix.csv',header=None,index=None)

# Print coding performance time
end = time()
print('Seconds elapsed: ' + str(end - start))