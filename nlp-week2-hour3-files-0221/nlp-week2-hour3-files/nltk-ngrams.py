import re
from nltk.util import ngrams

str = "Natural-language processing (NLP) is an area of computer science and artificial intelligence concerned with the interactions between computers and human (natural) languages."

str = str.lower()
str = re.sub(r'[^a-zA-Z0-9\s]', ' ', str)
tokens = [token for token in str.split(" ") if token != ""]
grams5 = list(ngrams(tokens, 5))

print("Generated 5-grams:")
print(grams5)


