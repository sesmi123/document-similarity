import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# nltk.download('omw-1.4')
# nltk.download('wordnet')
# nltk.download('stopwords')

# Initialize the WordNet lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(doc):
    # Convert to lower case
    doc = doc.lower()
    # Remove punctuation
    doc = re.sub(r'\W', ' ', doc)
    # Remove extra spaces
    doc = re.sub(r'\s+', ' ', doc).strip()
    # Lemmatization
    words = doc.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

