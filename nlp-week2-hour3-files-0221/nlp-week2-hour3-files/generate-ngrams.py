import re

str = "Natural-language processing (NLP) is an area of computer science and artificial intelligence concerned with the interactions between computers and human (natural) languages."

def generate_ngrams(s, n):
  # Convert to lowercases
  s = s.lower()
  
  # Replace all none alphanumeric characters with spaces
  s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)
  
  # Break sentence in the token, remove empty tokens
  tokens = [token for token in s.split(" ") if token != ""]
  
  # Use the zip function to help us generate n-grams
  # Concatentate the tokens into ngrams and return
  ngrams = zip(*[tokens[i:] for i in range(n)])
  return [" ".join(ngram) for ngram in ngrams]

grams2 = generate_ngrams(str, 2)
print("Generated 2-grams:")
print(grams2)
print("--------------------------")

grams5 = generate_ngrams(str, 5)
print("Generated 5-grams:")
print(grams5)


