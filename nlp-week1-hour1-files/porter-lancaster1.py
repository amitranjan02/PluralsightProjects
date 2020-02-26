from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer

# create two stemmers:
porter = PorterStemmer()
lancaster = LancasterStemmer()

#provide a word to be stemmed
print("Porter Stemmer:")
print(porter.stem("cats"))
print(porter.stem("trouble"))
print(porter.stem("troubling"))
print(porter.stem("troubled"))

print("Lancaster Stemmer:")
print(lancaster.stem("cats"))
print(lancaster.stem("trouble"))
print(lancaster.stem("troubling"))
print(lancaster.stem("troubled"))

