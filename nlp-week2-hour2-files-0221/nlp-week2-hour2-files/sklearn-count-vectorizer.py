from sklearn.feature_extraction.text import CountVectorizer

# create CountVectorizer object
vectorizer = CountVectorizer()

corpus = [
          'Text of first document.',
          'Text of the second document made longer.',
          'Number three.',
          'This is number four.',
]

# learn the vocabulary and store CountVectorizer matrix in X
# columns of X correspond to the result of this method
X = vectorizer.fit_transform(corpus)
print("Matrix X from CountVectorizer:")
print(X)

# retrieving the matrix in the numpy form
vectorizer.get_feature_names() == (
    ['document', 'first', 'four', 'is', 'longer',
     'made', 'number', 'of', 'second', 'text',
     'the', 'this', 'three'])

# transforming a new document according to learn vocabulary
X = X.toarray()
print("X as an array:")
print(X)

result = vectorizer.transform(['A new document.']).toarray()
print("result of document transformation:")
print(result)

