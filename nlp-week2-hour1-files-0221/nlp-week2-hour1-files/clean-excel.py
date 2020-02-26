# https://medium.com/@japneet121/document-vectorization-301b06a041

import pandas as pd
df = pd.read_csv('s_news.csv')

#drop the Nan rows
df.dropna(inplace=True)

import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import gensim

lemma = WordNetLemmatizer()

stopword_set = set(stopwords.words('english')+['a','of','at','s','for','share','stock'])

def process(string):
  string = ' '+string+' '
  string = ' '.join([word if word not in stopword_set else '' for word in string.split()])
  string = re.sub('\@\w*',' ',string)
  string = re.sub('\.',' ',string)
  string = re.sub("[,#'-\(\):$;\?%]",' ',string)
  string = re.sub("\d",' ',string)
  string = string.lower()
  string = re.sub("nyse",' ',string)
  string = re.sub("inc",' ',string)
  string = re.sub(r'[^\x00-\x7F]+',' ', string)
  string = re.sub(' for ',' ', string)
  string = re.sub(' s ',' ', string)
  string = re.sub(' the ',' ', string)
  string = re.sub(' a ',' ', string)
  string = re.sub(' with ',' ', string)
  string = re.sub(' is ',' ', string)
  string = re.sub(' at ',' ', string)
  string = re.sub(' to ',' ', string)
  string = re.sub(' by ',' ', string)
  string = re.sub(' & ',' ', string)
  string = re.sub(' of ',' ', string)
  string = re.sub(' are ',' ', string)
  string = re.sub(' co ',' ', string)
  string = re.sub(' stock ',' ', string)
  string = re.sub(' share ',' ', string)
  string = re.sub(' stake ',' ', string)
  string = re.sub(' corporation ',' ', string)
  string = " ".join(lemma.lemmatize(word) for word in string.split())
  string = re.sub('( [\w]{1,2} )',' ', string)
  string = re.sub("\s+",' ',string)
  
  return string.split()#drop the duplicate values of news 

df.drop_duplicates(subset = ['raw.title'],keep='last',inplace=True)#reindex the data frame

df.index = range(0,len(df))#apply the process function to the news titles

df['title_l'] = df['raw.title'].apply(process)
df_new = df

print("new data frame:")
print(df_new)

