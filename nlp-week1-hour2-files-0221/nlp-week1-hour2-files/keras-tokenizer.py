from keras.preprocessing.text import Tokenizer
import numpy as np

texts = ['The cat sat on the mat.',
         'The dog sat on the log.',
         'Dogs and cats living together.']
tokenizer = Tokenizer(num_words=10)
tokenizer.fit_on_texts(texts)

sequences = []
for seq in tokenizer.texts_to_sequences_generator(texts):
  sequences.append(seq)
assert np.max(np.max(sequences)) < 10
assert np.min(np.min(sequences)) == 1

tokenizer.fit_on_sequences(sequences)

for mode in ['binary', 'count', 'tfidf', 'freq']:
  matrix = tokenizer.texts_to_matrix(texts, mode) 

print("texts:",texts)
print("=> Found %s unique tokens <=" % len(tokenizer.word_index))

