from flask import Flask , request
from langdetect import detect
from googletrans import Translator
from nltk.tokenize import word_tokenize,RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import seaborn as sns
import matplotlib.pyplot as plt 
from string import punctuation 
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans,AgglomerativeClustering
from wordcloud import WordCloud
from collections import Counter
from sklearn.metrics import silhouette_score
from gensim.models import Word2Vec
from nltk.stem import LancasterStemmer,WordNetLemmatizer
from nltk.corpus import stopwords
from string import punctuation
import contractions 
import unidecode 
import re 
app = Flask(__name__)
@app.route("/")
def welcome():
    return "hey welcome to my web api"
@app.route("/result/<int:marks>")
def result(marks):
    if marks >45:
        return "the student is passed with marks"+ str(marks)
    else:
        return "The student has failed with marks"+ str(marks)
@app.route("/preprocessig/data")
def remove_newline(data):
    clean_text = data.replace("\\n",' ').replace("\n",' ').replace('\t',' ').replace('\\',' ')
    return clean_text
def remove_whitespace(data):
    pattern = re.compile(r'\s+')
    without_whitespace =  re.sub(pattern,' ',data)
    return without_whitespace
def remove_accented_character(data):
    text  = unidecode.unidecode(data)
    return text 
def contraction_mapping(data):
    tokens = data.split()
    expanded_words = []
    for word in tokens :
        expanded_words.append(contractions.fix(word))
    expanded_text = " ".join(expanded_words)
    return expanded_text
stop = stopwords.words('english')
stop.remove('no')
stop.remove('nor')
stop.remove(
 'not')
def clean_data(data):
    tokens = RegexpTokenizer(r'\w+').tokenize(data)
    text_without_stop = [word for word in tokens if word not in stop]
    final_text = []
    for word in text_without_stop :
        if (len(word)<2) or ( word in punctuation) :
            pass 
        else :
            final_text.append(word)
    text = " ".join(final_text)
    return text
def lemmatization(data):
    tokens = word_tokenize(data)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    lemma = WordNetLemmatizer()
    final_text =[]
    for i in tokens:
        lemmatized_word = lemma.lemmatize(i)
        final_text.append(lemmatized_word)
    return " ".join(final_text)
       
    
if __name__ == "__main__":
    app.run()
