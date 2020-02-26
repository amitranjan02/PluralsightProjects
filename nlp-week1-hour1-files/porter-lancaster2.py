from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer

# create two stemmers:
porter = PorterStemmer()
lancaster = LancasterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()


word_list = ["friend", "friendship", "friends", "friendships","stabil","destabilize","misunderstanding","railroad","moonlight","football"]

print("{0:20}{1:20}{2:20}{3:20}".format("Word","Porter Stemmer","lancaster Stemmer","wordnet Stemmer"))

for word in word_list:
  print("{0:20}{1:20}{2:20}{3:20}".format(word,porter.stem(word),lancaster.stem(word),wordnet_lemmatizer.lemmatize(word, pos = 'v') ))


