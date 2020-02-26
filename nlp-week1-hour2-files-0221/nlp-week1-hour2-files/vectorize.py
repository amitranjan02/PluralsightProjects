# https://nlpforhackers.io/deep-learning-introduction/

VOCAB = ['dog', 'cheese', 'cat', 'mouse']
TEXT1 = 'the mouse ate the cheese'
TEXT2 = 'the horse ate the hay'
 
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(vocabulary=VOCAB)

result1 = vectorizer.transform([TEXT1]).todense() # matrix([[0,1,0,1]])
result2 = vectorizer.transform([TEXT2]).todense() # matrix([[0,1,0,1]])

print("VOCAB: ",VOCAB)
print("TEXT1:",TEXT1)  
print("BOW1: ",result1)  # [0, 1, 0, 1]
print("")

print("TEXT2:",TEXT2)  
print("BOW2: ",result2)  # [0, 1, 0, 1]

