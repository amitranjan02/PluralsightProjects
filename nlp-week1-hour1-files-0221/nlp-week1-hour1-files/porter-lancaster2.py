from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer

# create two stemmers:
porter = PorterStemmer()
lancaster = LancasterStemmer()

word_list = ["friend", "friendship", "friends", "friendships","stabil","destabilize","misunderstanding","railroad","moonlight","football"]

print("{0:20}{1:20}{2:20}".format("Word","Porter Stemmer","lancaster Stemmer"))

for word in word_list:
  print("{0:20}{1:20}{2:20}".format(word,porter.stem(word),lancaster.stem(word)))

