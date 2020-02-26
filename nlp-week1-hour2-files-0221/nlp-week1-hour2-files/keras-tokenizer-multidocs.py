from keras.preprocessing.text import Tokenizer

# define 5 documents
docs = ['Well done!',
        'Good work',
        'Great effort',
        'nice work',
        'Excellent!']

# create the tokenizer
t = Tokenizer()

# fit the tokenizer on the documents
t.fit_on_texts(docs)

# summarize what was learned
print("word counts:",t.word_counts)
print("doc  counts:",t.document_count)
print("word index: ",t.word_index)
print("word docs:  ",t.word_docs)

