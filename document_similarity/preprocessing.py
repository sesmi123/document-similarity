import re # regular expressions
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Initialize the WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words('english'))

def preprocess(doc):

    # Convert to lower case
    doc = doc.lower()

    # Remove punctuation
    doc = re.sub(r'\W', ' ', doc)

    # Remove extra spaces
    doc = re.sub(r'\s+', ' ', doc).strip()

    # Lemmatization and remove stop words
    words = word_tokenize(doc)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return ' '.join(words)

