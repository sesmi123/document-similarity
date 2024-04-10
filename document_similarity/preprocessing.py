import re # regular expressions
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class DocumentPreProcessor:

    def __init__(self) -> None:
        # Initialize the WordNet lemmatizer
        self.lemmatizer = WordNetLemmatizer()

        self.stop_words = set(stopwords.words('english'))

    def preprocess(self, doc):

        # Convert to lower case
        doc = doc.lower()

        # Remove punctuation
        doc = re.sub(r'\W', ' ', doc)

        # Remove extra spaces
        doc = re.sub(r'\s+', ' ', doc).strip()

        # Lemmatization and remove stop words
        words = word_tokenize(doc)
        words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]
        
        return ' '.join(words)

