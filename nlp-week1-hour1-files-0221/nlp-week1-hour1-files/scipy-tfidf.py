import scipy as sp
def tfidf(term, doc, corpus):
  tf = doc.count(term) / len(doc)
  num_docs_with_term = len([d for d in corpus if term in d])
  idf = sp.log(len(corpus) / num_docs_with_term)
  return tf * idf

a, abb, abc = ["a"], ["a", "b", "b"], ["a", "b", "c"]
D = [a, abb, abc]
print(tfidf("a", a, D))
#0.0
print(tfidf("a", abb, D))
#0.0
print(tfidf("a", abc, D))
#0.0
print(tfidf("b", abb, D))
#0.270310072072
print(tfidf("a", abc, D))
#0.0
print(tfidf("b", abc, D))
#0.135155036036
print(tfidf("c", abc, D))
#0.366204096223

