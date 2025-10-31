# ----- Cell 1 -----
import pandas as pd
import numpy as np

import re  # for working with regular expressions
import nltk  # for natural language processing (nlp)
try:
    import spacy  # also for nlp
except Exception:
    spacy = None
    print("spaCy not available. Some functionality may be limited.")
import string

# ----- Cell 2 -----
import sys
import subprocess

try:
    import tensorflow as tf
except ImportError:
    print("TensorFlow not found. Installing tensorflow...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow"])
    import tensorflow as tf

print("TF:", tf.__version__)
try:
    print("GPUs:", tf.config.list_physical_devices('GPU'))
except Exception as e:
    print("Could not list GPUs:", e)

# ----- Cell 3 -----
trdf = pd.read_csv('train.csv', header='infer')

print(trdf.head(3))

trdf.info()

# ----- Cell 4 -----
try:
    print(trdf.value_counts())
except Exception:
    # value_counts may not be supported directly for multi-column DataFrame in some pandas versions
    print("Could not compute value_counts on the dataframe directly. Consider specifying a column.")

# ----- Cell 5 -----
trdf['lowered_text'] = trdf['text'].str.lower()
print(trdf.shape)
# confirm the case conversion
print(trdf['lowered_text'].head(3))

# ----- Cell 6 -----
punc = string.punctuation

print(type(punc))
print(punc)

# ----- Cell 7 -----
trans_map = str.maketrans("", "", punc)

print(type(trans_map))
print(trans_map)

# ----- Cell 8 -----
print(trdf['lowered_text'].head(10))
trdf['lowered_text_without_punc'] = trdf["lowered_text"].str.translate(trans_map)
print(trdf['lowered_text_without_punc'].head(10))

# ----- Cell 9 -----
def remove_punctuation(in_str):
    return in_str.translate(trans_map)

print(trdf['lowered_text'].head(10))
trdf['lowered_text_without_punc_2'] = trdf["lowered_text"].apply(remove_punctuation)
print(trdf['lowered_text_without_punc_2'].head(10))

# ----- Cell 10 -----
# nltk.download("all")
# nltk.download("stopwords")
from nltk.corpus import stopwords

# Ensure necessary NLTK data is available
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

print(type(stopwords.words()), len(stopwords.words()))
# type is list of strings/words
# stopwords.words()  # Very huge list (7702 stopwords) as it includes stopwords from many languages

# ----- Cell 11 -----
print(type(stopwords.words('english')), len(stopwords.words('english')))  # list, 179 stopwords

# now fetch stopwords in English only
stopwords_eng = list(stopwords.words('english'))
stopwords_eng.sort()
print(stopwords_eng)

# ----- Cell 12 -----
print(trdf["lowered_text"].head(10))

def remove_stopwords(in_str):
    new_str = ''
    words = str(in_str).split()
    for tx in words:
        if tx not in stopwords_eng:
            new_str = new_str + tx + " "
    return new_str

trdf['lowered_text_stop_removed'] = trdf["lowered_text"].apply(remove_stopwords)

print(trdf["lowered_text_stop_removed"].head(10))

# ----- Cell 13 -----
text = input("Enter text :")

sentence = nltk.sent_tokenize(text)
print("Sentences: ", sentence)

word = nltk.word_tokenize(text)
print("words", word)

# ----- Cell 14 -----
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Initialize stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def stem_text(text):
    return ' '.join([stemmer.stem(word) for word in str(text).split()])

def lemmatize_text(text):
    return ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(str(text))])

trdf['stemmed'] = trdf['lowered_text_stop_removed'].apply(stem_text)

trdf['lemmatized'] = trdf['lowered_text_stop_removed'].apply(lemmatize_text)

print(trdf[['lowered_text_stop_removed', 'stemmed', 'lemmatized']].head())

# ----- Cell 15 -----
print(trdf.columns)

# ----- Cell 16 -----
# This cell intentionally left blank