import re
import string
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
tf = pickle.load(open("artifacts/tf.pkl","rb"))
stop_words = stopwords.words('english')
negative_verbs = ['not', 'no', 'doesnt', 'dont', 'didnt', 'wasnt', 'werent', 'hasnt', 'havent', 'isnt', 'arent', 'werent']

stop_words = [word for word in stop_words if word not in negative_verbs]
def preproccessing(text):
    text = text.lower()
   
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    text = re.sub("[^a-z]", ' ', text)
    
    
    tokens_text = word_tokenize(text)
    
    
    tokens_text = [word for word in tokens_text if word not in stop_words]
    
    
    stemmer = PorterStemmer()
    tokens_text = [stemmer.stem(word) for word in tokens_text]
    text = ' '.join(tokens_text)
    text = tf.transform([text])
    return text


