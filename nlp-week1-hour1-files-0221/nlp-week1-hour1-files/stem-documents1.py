from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

# read file contents:
file = open("data-science-wiki.txt")
my_lines_list = file.readlines()
#print("my_lines_list:")
#print(my_lines_list)

porter = PorterStemmer()

def stemSentence(sentence):
  token_words=word_tokenize(sentence)
  token_words
  stem_sentence = []

  for word in token_words:
    stem_sentence.append(porter.stem(word))
    stem_sentence.append(" ")

  return "".join(stem_sentence)

def saveStemmedLines():
  stem_file = open("stem-data-science-wiki.txt",mode="a+",encoding="utf-8")
  for line in my_lines_list:
    stem_sentence = stemSentence(line)
    stem_file.write(stem_sentence)
  stem_file.close()

print(my_lines_list[0])
print("Stemmed sentence:")
x = stemSentence(my_lines_list[0])
print(x)

saveStemmedLines()

