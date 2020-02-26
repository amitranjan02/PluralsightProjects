from keras.preprocessing.text import Tokenizer

# define five documents:
docs = ['Well done!',
        'Good work',
        'Great effort',
        'nice work',
        'Excellent!']

# create and then fit the tokenizer
t = Tokenizer()
t.fit_on_texts(docs)

# summarize what was learned
print("word counts:",t.word_counts)
print("doc  counts:",t.document_count)
print("word index: ",t.word_index)
print("word docs:  ",t.word_docs)

# integer encode documents
encoded_docs = t.texts_to_matrix(docs, mode='count')
print("encoded_docs:")
print(encoded_docs)

