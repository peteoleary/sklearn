import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
# If nltk stop word is not downloaded
# nltk.download('stopwords')

# List of documents
a1 = "He is a good dog."
a2 = "The dog is too lazy."
a3 = "That is a brown cat."
a4 = "The cat is very active."
a5 = "I have brown cat and dog."

df = pd.DataFrame()
df["documents"] = [a1,a2,a3,a4,a5]

# Preprocessing
df['clean_documents'] = df['documents'].str.replace("[^a-zA-Z#]", " ")
df['clean_documents'] = df['clean_documents'].fillna('').apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))
df['clean_documents'] = df['clean_documents'].fillna('').apply(lambda x: x.lower())

# tokenization
tokenized_doc = df['clean_documents'].fillna('').apply(lambda x: x.split())

# de-tokenization
detokenized_doc = []
for i in range(len(df)):
    t = ' '.join(tokenized_doc[i])
    detokenized_doc.append(t)

df['clean_documents'] = detokenized_doc

# TF-IDF vector
vectorizer = TfidfVectorizer(stop_words='english', smooth_idf=True)
X = vectorizer.fit_transform(df['clean_documents'])

# SVD represent documents and terms in vectors 
svd_model = TruncatedSVD(n_components=2, algorithm='randomized', n_iter=100, random_state=122)
lsa = svd_model.fit_transform(X)

# Documents - Topic vector
pd.options.display.float_format = '{:,.16f}'.format
topic_encoded_df = pd.DataFrame(lsa, columns = ["topic_1", "topic_2"])
topic_encoded_df["documents"] = df['clean_documents']

topic_encoded_df.describe()

# display(topic_encoded_df[["documents", "topic_1", "topic_2"]])

# Features or words used as features 
dictionary = vectorizer.get_feature_names()

# Term-Topic matrix
encoding_matrix = pd.DataFrame(svd_model.components_, index = ["topic_1","topic_2"], columns = (dictionary)).T