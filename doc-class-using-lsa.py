from gift_basket_data import X, vectorizer

from sklearn.decomposition import TruncatedSVD
lsa = TruncatedSVD(n_components=40,n_iter=100)

lsa.fit(X)

terms = vectorizer.get_feature_names()

for i,comp in enumerate(lsa.components_):
    termsInComp = zip(terms,comp)
    sortedterms = sorted(termsInComp, key=lambda x: x[1],reverse=True)[:10]
    print("Concept %d:" % i)
    for term in sortedterms:
        print(term[0])
    print(" ")
