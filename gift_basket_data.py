import csv
import re
import string

# TODO: add the part number in here somehow?
input_lines = []
input_part_nums = []

with open('vip_gifts_and_baskets products 2020-04-23T0935.csv') as csvfile:
    reader = csv.DictReader(csvfile)

    for row in reader:
        input_lines.append(row['Products Full Description'] + ' ' + row['Products Contents'])
        input_part_nums.append(row['Products Part Number'])


from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

import nltk
from nltk.stem import PorterStemmer
# init stemmer
porter_stemmer = PorterStemmer()
 
def my_cool_preprocessor(text):
    
    text = text.lower() 
    text = re.sub('<.*?>', '', text)
    text = re.sub("\\W"," ",text) # remove special chars
    
    # stem words
    words = re.split("\\s+",text)
    # words = list(filter(lambda x: (not (x.isnumeric())), words))
    stemmed_words = [porter_stemmer.stem(word=word) for word in words]
    return ' '.join(stemmed_words)

gifts_vectorizer = CountVectorizer(input_lines, ngram_range=(1,2), max_df=0.3, preprocessor=my_cool_preprocessor)
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
  print("VOCABULARY")
  print(df_vocabulary)
  print("STOP WORDS")
  print(df_stop_words)
  print("IDF")
  print(df_idf)