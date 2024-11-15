#UCSD FALL 2023
#CSE 251U
#Homework 4.6

#Name: Arjun H. Badami
#PID: A13230476

import json
from collections import defaultdict

import numpy as np
import sklearn
from sklearn import linear_model
import numpy
import random
import gzip
import math
import random
from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import brown
from nltk.corpus import stopwords
from numpy.linalg import norm
from scipy.cluster import hierarchy

stops = stopwords.words('english')
allwords = brown.words()
puncs = ['.', ',', '?', '!', '#', '%', '/', '*', '``', "''", ';', '--', 'one', 'would', ')', '(', 'said', ':', 'new',
         'could', 'time', 'two', 'may', 'first', 'like', 'man', 'even', 'made', 'also', 'many', 'must', 'af', 'back',
         'years', 'much', 'way', 'well', 'people', 'mr.', 'little', 'state', 'good', 'make', 'world', 'still', 'see',
         'men', 'work', 'long', 'get', 'life', 'never', 'day', 'another', 'know', 'last', 'us', 'might', 'great', 'old',
         'year', 'come', 'since', 'go', 'came', 'right', 'used', 'three', 'take', 'states', 'use', 'house', 'without',
         'place']
Vocab = defaultdict(int)

for word in allwords:
    w = word.strip().lower()
    if(w not in stops and w not in puncs and '$' not in w):
        Vocab[w] += 1


all_word_counts = []
for w in Vocab:
    all_word_counts.append((Vocab[w], w))

all_word_counts.sort(reverse=True)

V = [t[1] for t in all_word_counts[:5000]]
C = [t[1] for t in all_word_counts[:1000]]

wordstoindices = {}
indicestowords = {}
for i in range(len(V)):
    wordstoindices[V[i]] = i
    indicestowords[i] = V[i]


N = [[0]*len(C) for _ in range(len(V))]

for i in range(len(allwords)):
    word = allwords[i]
    if word in V:
        w = wordstoindices[word]
        if (i - 2 >= 0 and allwords[i-2] in C):
            c = wordstoindices[allwords[i-2]]
            N[w][c] += 1
        if (i - 1 >= 0 and allwords[i-1] in C):
            c = wordstoindices[allwords[i-1]]
            N[w][c] += 1
        if (i + 1 < len(allwords) and allwords[i+1] in C):
            c = wordstoindices[allwords[i+1]]
            N[w][c] += 1
        if (i + 2 < len(allwords) and allwords[i+2] in C):
            c = wordstoindices[allwords[i+2]]
            N[w][c] += 1


P_c_w = [[0]*len(C) for _ in range(len(V))]
P_c = [0]*len(C)
total = 0
for w in range(len(V)):
    w_sum = sum(N[w])
    if w_sum == 0:
        continue
    for c in range(len(C)):
        P_c_w[w][c] = N[w][c] / w_sum
        total += N[w][c]


for c in range(len(C)):
    P_c[c] = sum([w[c] for w in N]) / total


Phi = [[0]*len(C) for _ in range(len(V))]
for w in range(len(V)):
    for c in range(len(C)):
        if P_c[c] > 0 and P_c_w[w][c] > 0:
            Phi[w][c] = max(0, math.log((P_c_w[w][c] / P_c[c]), 10))


Phi = np.array(Phi)
svd = TruncatedSVD(n_components=100)
Embeddings = svd.fit_transform(Phi)

def cosine_sim(A, B):
    return (1 - (np.dot(A,B)/(norm(A)*norm(B))))


def find_nn(i):
    dist = math.inf
    nn = -1
    A = Embeddings[i]
    for j in range(len(Embeddings)):
        if j == i:
            continue
        B = Embeddings[j]
        d = cosine_sim(A, B)
        if (d < dist):
            dist = d
            nn = j

    return indicestowords[nn]


investigate_words = ['school', 'water', 'system', 'government', 'eyes', 'national', 'children', 'church', 'power', 'family',
                     'mind', 'country', 'service', 'god', 'certain', 'law', 'human', 'company', 'local', 'history', 'action',
                     'feet', 'death', 'experience', 'body']


for word in investigate_words:
    print('WORD:    ' + word)
    nn = find_nn(wordstoindices[word])
    print('NEAREST NEIGHBOR:    ' + nn)
    print('=====================================================')


Z = hierarchy.linkage(Embeddings, 'ward')
groups = hierarchy.fcluster(Z, 100, criterion='maxclust')
clusters = defaultdict(list)
for idx in range(len(groups)):
    group_id = groups[idx]
    name = V[idx]
    clusters[group_id].append(name)

print(clusters)
