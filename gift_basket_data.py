import csv
import re
import string

def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext

def cleanpunc(t):
    cleanr = re.compile('[\n\.,;]')
    cleantext = re.sub(cleanr, ' ', t)
    return cleantext

def clean_one_value(v):
   v = cleanhtml(v)
   v = cleanpunc(v)
   v = remove_unused_words(v)
   # TODO: augment with word stemming, plurals, synonyms?
   return v

def filter_fun(v):
  if v.isnumeric():
    return False
  if (v in ['oz', 'ml', 'gift', 'basket', 'includes']):
    return False
  return True


def remove_unused_words(v):
  words = v.split()
  words = filter(filter_fun, words)
  return ' '.join(list(words))

# TODO: add the part number in here somehow?
input_lines = []
input_part_nums = []

with open('vip_gifts_and_baskets products 2020-04-23T0935.csv') as csvfile:
    reader = csv.DictReader(csvfile)

    for row in reader:
        input_lines.append(clean_one_value(row['Products Full Description']) + ' ' + clean_one_value(row['Products Contents']))
        input_part_nums.append(row['Products Part Number'])

from sklearn.feature_extraction.text import TfidfVectorizer
gifts_vectorizer = TfidfVectorizer(stop_words = 'english')

gift_features = gifts_vectorizer.fit_transform(input_lines)