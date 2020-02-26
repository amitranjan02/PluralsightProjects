from sklearn.feature_extraction.text import TfidfTransformer

# create tf-idf object
transformer = TfidfTransformer(smooth_idf=False)

# X can be obtained as X.toarray() 
X = [[3, 0, 1],
     [5, 0, 0],
     [3, 0, 0],
     [1, 0, 0],
     [3, 2, 0],
     [3, 0, 4]]

# learn the vocabulary and store tf-idf sparse matrix in tfidf
tfidf = transformer.fit_transform(X)

# retrieving matrix in numpy form 
result = tfidf.toarray()                  

print("tfidf as an array:")
print(result)

