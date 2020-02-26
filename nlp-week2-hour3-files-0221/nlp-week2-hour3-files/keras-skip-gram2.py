import tensorflow as tf

#--------------------------
import re
import numpy as np
import nltk
from nltk.corpus import gutenberg
from string import punctuation

wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')

def normalize_document(doc):
  # lower case and remove special characters\whitespaces
  doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I|re.A)
  doc = doc.lower()
  doc = doc.strip()
  # tokenize document
  tokens = wpt.tokenize(doc)
  # filter stopwords out of document
  filtered_tokens = [token for token in tokens if token not in stop_words]
  # re-create document from filtered tokens
  doc = ' '.join(filtered_tokens)
  return doc

normalize_corpus = np.vectorize(normalize_document)

bible = gutenberg.sents('bible-kjv.txt') 
remove_terms = punctuation + '0123456789'

norm_bible = [[word.lower() for word in sent if word not in remove_terms] for sent in bible]
norm_bible = [' '.join(tok_sent) for tok_sent in norm_bible]
norm_bible = filter(None, normalize_corpus(norm_bible))
norm_bible = [tok_sent for tok_sent in norm_bible if len(tok_sent.split()) > 2]

print('Total lines:', len(bible))
print('\nSample line:', bible[10])
print('\nProcessed line:', norm_bible[10])
#--------------------------

tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(text)

word2id = tokenizer.word_index
id2word = {v:k for k, v in word2id.items()}

vocab_size = len(word2id) + 1 
embed_size = 100

wids = [[word2id[w] for w in tf.keras.preprocessing.text.text_to_word_sequence(doc)] for doc in norm_bible]

print('wids:', wids)
print('Vocabulary Size:', vocab_size)
print('Vocabulary Sample:', list(word2id.items())[:10])

# generate skip-grams
skip_grams = [tf.keras.preprocessing.sequence.skipgrams(wid, vocabulary_size=vocab_size, window_size=10) for wid in wids]

print("=====> skip_grams:")
print(skip_grams)
print("skip_grams[0][0]:")
print(skip_grams[0][0])
print("skip_grams[0][1]:")
print(skip_grams[0][1])

# view sample skip-grams
pairs, labels = skip_grams[0][0], skip_grams[0][1]

for i in range(10):
  print("({:s} ({:d}), {:s} ({:d})) -> {:d}".format(
    id2word[pairs[i][0]], pairs[i][0], 
    id2word[pairs[i][1]], pairs[i][1], 
    labels[i]))

