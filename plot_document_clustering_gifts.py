# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Lars Buitinck
# License: BSD 3 clause
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import MiniBatchKMeans

import logging
from optparse import OptionParser
import sys
from time import time

import numpy as np


# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

n_reduced_dimensions = 100

from gift_basket_data import gift_features, gifts_vectorizer, gifts_tfidf, input_lines, input_part_nums

true_k = 20

print("n_samples: %d, n_features: %d" % gift_features.shape)
print()

print("Performing dimensionality reduction using LSA")
t0 = time()
# Vectorizer results are normalized, which makes KMeans behave as
# spherical k-means for better results. Since LSA/SVD results are
# not normalized, we have to redo the normalization.
svd = TruncatedSVD(n_reduced_dimensions)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)

# fit_transform reduces dimensions
gift_features_reduced = lsa.fit_transform(gift_features)

print("done in %fs" % (time() - t0))

explained_variance = svd.explained_variance_ratio_.sum()
print("Explained variance of the SVD step: {}%".format(
    int(explained_variance * 100)))

print()

km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                        init_size=1000, batch_size=1000, verbose=False)

print("Clustering sparse data with %s" % km)
t0 = time()
km.fit(gift_features_reduced)
print("done in %0.3fs" % (time() - t0))
print()

print("Top terms per cluster:")

# turn the reduced clusters back into their original vectors
original_space_centroids = svd.inverse_transform(km.cluster_centers_)

# get the indexes which would sort the clusters, this gives us an index into the original terms
order_centroids = original_space_centroids.argsort()

# do a left right swap so each vector is in descending order by value
order_centroids = order_centroids[:, ::-1]

# get the original terms
terms = gifts_vectorizer.get_feature_names()

all_label_terms = []

for i in range(true_k):
    label_terms = ''
    for ind in order_centroids[i, :20]:
        label_terms += ' %s' % terms[ind]
    print('%d: %s' % (i, label_terms))
    all_label_terms.append(label_terms)


print()

# find distance between all rows of gift_features_reduced and km.cluster_centers_
kernel_matrix = metrics.pairwise.cosine_similarity(gift_features_reduced, km.cluster_centers_, dense_output=True)

# TODO: check to make sure length(input_part_nums) == gift_features_reduced.shape[0]

# n = len(input_part_nums)
n = 10
for i in range(n):
    sorted_similarities = kernel_matrix[i].argsort()
    sorted_similarities = sorted_similarities[::-1]
    sorted_similarities_string = ''
    for j in sorted_similarities[0:5]:
        sorted_similarities_string += '%d, ' % j
    print('%s: %s' % (input_part_nums[i], sorted_similarities_string))