import csv
import re
import string

def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext

def clean_one_value(v):
   v = cleanhtml(v)
   # TODO: augment with word stemming, plurals, synonyms?
   return v

# TODO: add the part number in here somehow?
input_lines = []
input_part_nums = []

with open('vip_gifts_and_baskets products 2020-04-23T0935.csv') as csvfile:
    reader = csv.DictReader(csvfile)

    for row in reader:
        input_lines.append(clean_one_value(row['Products Full Description']) + ' ' + clean_one_value(row['Products Contents']))
        input_part_nums.append(row['Products Part Number'])

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

gifts_vectorizer = CountVectorizer(input_lines, max_df=0.50, min_df=5)
word_count_vector=gifts_vectorizer.fit_transform(input_lines)

import pandas as pd

# DataFrames for inspection purposes...
df_vocabulary = pd.DataFrame(word_count_vector.toarray().sum(axis=0), gifts_vectorizer.get_feature_names())
df_stop_words = pd.DataFrame(gifts_vectorizer.stop_words_)

gifts_tfidf = TfidfTransformer()

gift_features = gifts_tfidf.fit_transform(word_count_vector)


df_idf = pd.DataFrame(gifts_tfidf.idf_, index=gifts_vectorizer.get_feature_names(),columns=["idf_weights"])
 
# sort ascending
df_idf.sort_values(by=['idf_weights'])

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
  print(df_vocabulary)
  # print(df_stop_words)
  # print(df_idf)