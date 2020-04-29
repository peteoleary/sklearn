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

cv = CountVectorizer(input_lines, max_df=0.50, min_df=5)
word_count_vector=cv.fit_transform(input_lines)

import pandas as pd

df_vocabulary = pd.DataFrame(word_count_vector.toarray().sum(axis=0), cv.get_feature_names())

gifts_vectorizer = TfidfTransformer()

gift_features = gifts_vectorizer.fit(word_count_vector)

df_stop_words = pd.DataFrame(cv.stop_words_)

df_idf = pd.DataFrame(gifts_vectorizer.idf_, index=cv.get_feature_names(),columns=["idf_weights"])
 
# sort ascending
df_idf.sort_values(by=['idf_weights'])

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
  print(df_vocabulary)
  # print(df_stop_words)
  # print(df_idf)